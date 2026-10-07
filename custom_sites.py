#!/usr/bin/env python3

"""Write the site-specific half of an air quality run: sites.yml.

The workflow itself never names a scheduler. Its jobs state cores, memory
and a wall-clock runtime, and the LSTM training job carries the Pegasus tag
"train". What the execution site is (an HTCondor pool, a Slurm cluster
reached through glite), which partition and account jobs land on, where
scratch lives, and what "train" means on that site is the site catalog's
business, and this module writes that catalog.

Precedence, most specific first:

  1. A sites.yml you wrote (by hand or with this script) — kept as-is.
  2. A hosted catalog named in ~/.pegasusrc (pegasus.catalog.site.repo.file),
     which Pegasus merges a local sites.yml over, key by key.
  3. A default written here: an HTCondor site, so the workflow plans with no
     setup at all (Pegasus Studio's default).

Only the requested site's entry is ever written; other sites in sites.yml are
left alone. A "local" site (submit-host scratch and output storage under the
workflow directory) is always ensured, because planning with "-o local" needs
it.

workflow_generator.py calls ensure_sites_yml() on every run (exposed there as
--site-style, --queue, --project, ...), so a Studio form or a shell gets the
same behaviour. Standalone use:

    # Unity or another cluster with a hosted catalog — add your account:
    ./custom_sites.py --style slurm --project my_lab

    # A local HTCondor pool with no hosted catalog:
    ./custom_sites.py --style condor --full

    # A Slurm cluster with no hosted catalog, training on a bigger partition:
    ./custom_sites.py --style slurm --full --queue cpu --project my_lab \\
        --scratch /scratch/$USER/airquality --train pegasus:queue=cpu-long
"""

import argparse
import io
import os
from pathlib import Path

import yaml
from Pegasus.api import (
    Directory, FileServer, Namespace, Operation, Site, SiteCatalog,
)

STYLES = ("condor", "slurm")

# Tag carried by the LSTM training job: the one long, memory-heavy step, so the
# one a site most plausibly wants on its own partition or with more time.
TRAIN_TAG = "train"

NAMESPACES = {ns.value: ns for ns in Namespace}


def parse_profile(text):
    """'ns:key=value' or 'key=value' (pegasus namespace) -> (ns, key, value)."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected NS:KEY=VALUE or KEY=VALUE")
    ns, colon, bare = key.partition(":")
    if not colon:
        ns, bare = "pegasus", key
    if ns not in NAMESPACES:
        raise argparse.ArgumentTypeError(
            f"{text!r}: unknown namespace {ns!r}; one of "
            f"{', '.join(sorted(NAMESPACES))}")
    return ns, bare, value


def hosted_catalog(pegasusrc=None):
    """The hosted catalog named in ~/.pegasusrc, or None."""
    rc = Path(pegasusrc) if pegasusrc else Path.home() / ".pegasusrc"
    name = None
    try:
        for line in rc.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "pegasus.catalog.site.repo.file":
                name = value.strip()
    except OSError:
        return None
    return name or None


def build_compute_site(name, style, full=True, queue=None, project=None,
                       scratch=None, storage=None, profiles=(), train=()):
    """A Site for `name`. `full=False` writes only the overrides (an overlay).

    Raises ValueError for combinations that cannot work.
    """
    if (scratch or storage) and not (full and style == "slurm"):
        raise ValueError("scratch/storage only apply to a full slurm site; a "
                         "condor pool stages through the submit host")
    site = Site(name)
    if full:
        if style == "slurm":
            if not queue:
                raise ValueError("a full slurm site needs a queue (the "
                                 "partition jobs submit to)")
            scratch = os.path.abspath(scratch or "work")
            storage = os.path.abspath(storage or "storage")
            site.add_directories(
                Directory(Directory.SHARED_SCRATCH, scratch,
                          shared_file_system=False)
                .add_file_servers(FileServer("file://" + scratch,
                                             Operation.ALL)),
                Directory(Directory.LOCAL_STORAGE, storage,
                          shared_file_system=False)
                .add_file_servers(FileServer("file://" + storage,
                                             Operation.ALL)),
            )
            site.add_condor_profile(grid_resource="batch slurm")
            site.add_pegasus_profile(style="glite",
                                     data_configuration="nonsharedfs",
                                     auxillary_local="true")
        else:
            site.add_condor_profile(universe="vanilla")
            site.add_pegasus_profile(style="condor")
    else:
        # An overlay still states how the site submits — the same values the
        # hosted entry carries, so merging changes nothing — so later runs
        # can tell (e.g. for bypass staging) without the hosted file at hand.
        if style == "slurm":
            site.add_condor_profile(grid_resource="batch slurm")
            site.add_pegasus_profile(style="glite")
        else:
            site.add_pegasus_profile(style="condor")
    if queue:
        site.add_pegasus_profile(queue=queue)
    if project:
        site.add_pegasus_profile(project=project)
    for ns, key, value in profiles:
        site.add_profiles(NAMESPACES[ns], key=key, value=value)
    # Tag profiles go in by hand (write_sites), not via Site.add_tag_profiles,
    # which only exists in Pegasus >= 5.1.3dev; the planner that reads them
    # has to be that new anyway, but generating should not.
    site.x_tags = {}
    for ns, key, value in train:
        site.x_tags.setdefault(TRAIN_TAG, {}).setdefault(ns, {})[key] = value
    return site


def build_local_site(wf_dir):
    """Submit-host scratch and output storage under the workflow directory."""
    scratch = os.path.join(wf_dir, "scratch")
    storage = os.path.join(wf_dir, "output")
    return Site("local").add_directories(
        Directory(Directory.SHARED_SCRATCH, scratch)
        .add_file_servers(FileServer("file://" + scratch, Operation.ALL)),
        Directory(Directory.LOCAL_STORAGE, storage)
        .add_file_servers(FileServer("file://" + storage, Operation.ALL)),
    )


def _site_dicts(*sites):
    """Pegasus Site objects -> the plain dicts a sites.yml holds."""
    sc = SiteCatalog()
    sc.add_sites(*sites)
    buf = io.StringIO()
    sc.write(buf, _format="yml")
    return yaml.safe_load(buf.getvalue())


def load_sites_yml(path):
    """{site name: entry} from a sites.yml; empty if it does not exist."""
    if not os.path.isfile(path):
        return None, {}
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    return doc, {s.get("name"): s for s in doc.get("sites") or []}


def site_style(entry):
    """'condor', 'slurm', another 'batch:<lrms>', or None if not stated."""
    profiles = (entry or {}).get("profiles") or {}
    style = (profiles.get("pegasus") or {}).get("style")
    grid = str((profiles.get("condor") or {}).get("grid_resource") or "")
    if style == "glite" or grid.startswith("batch"):
        lrms = grid.split()[1] if len(grid.split()) > 1 else "unknown"
        return "slurm" if lrms == "slurm" else f"batch:{lrms}"
    if style == "condor":
        return "condor"
    return None


def write_sites(path, sites, base_doc=None):
    """Write `sites` into path, replacing same-named entries, keeping others."""
    new = _site_dicts(*sites)
    for site, entry in zip(sites, new["sites"]):
        tags = getattr(site, "x_tags", None)
        if tags:
            entry["x-tags"] = [{"name": name, "profiles": profiles}
                               for name, profiles in tags.items()]
    doc = base_doc or new
    by_name = {s["name"]: s for s in new["sites"]}
    kept = [s for s in doc.get("sites") or [] if s.get("name") not in by_name]
    doc["sites"] = kept + new["sites"]
    doc.setdefault("pegasus", new.get("pegasus"))
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)


def _hosted_style(path, hosted, site_name):
    """The site's style per the hosted catalog, or None if unknown.

    The planner leaves a copy of the hosted file in the directory it ran
    in; without that copy (first run) the style cannot be read.
    """
    if not hosted:
        return None
    _, entries = load_sites_yml(
        os.path.join(os.path.dirname(os.path.abspath(path)), hosted))
    return site_style(entries.get(site_name))


def ensure_sites_yml(path, site_name, wf_dir, style="auto", **site_opts):
    """Make sure planning against `site_name` works; return (action, style).

    style "auto" keeps anything already provided and only fills gaps;
    "condor"/"slurm" (re)write the site_name entry; "none" writes nothing.
    `action` says what happened, for the caller to report.
    """
    if style == "none":
        _, entries = load_sites_yml(path)
        return "untouched (--site-style none)", site_style(
            entries.get(site_name))

    doc, entries = load_sites_yml(path)
    to_write = []
    if "local" not in entries:
        to_write.append(build_local_site(wf_dir))

    hosted = hosted_catalog()
    if style in STYLES:
        if site_name == "local":
            raise ValueError("--site-style describes the execution site; it "
                             "cannot be applied to 'local'")
        # Over a hosted catalog the hosted entry already carries the
        # scheduler settings, so write only the overrides.
        full = site_opts.pop("full", None)
        if full is None:
            full = not hosted
        to_write.append(build_compute_site(site_name, style, full=full,
                                           **site_opts))
        if not full and (_hosted_style(path, hosted, site_name)
                         not in (None, style)):
            raise ValueError(
                f"--site-style {style} contradicts hosted catalog {hosted}, "
                f"where {site_name!r} is "
                f"{_hosted_style(path, hosted, site_name)}")
        action = (f"wrote {style} site {site_name!r}"
                  + ("" if full else " (overlay)"))
        effective = style
    elif site_name in entries or site_name == "local":
        action = f"using existing site {site_name!r}"
        # An overlay on a hosted catalog states no style of its own.
        effective = (site_style(entries.get(site_name))
                     or _hosted_style(path, hosted, site_name))
    elif hosted:
        action = (f"site {site_name!r} expected from hosted catalog {hosted}")
        effective = _hosted_style(path, hosted, site_name)
    else:
        site_opts.pop("full", None)
        to_write.append(build_compute_site(site_name, "condor", full=True,
                                           **site_opts))
        action = f"added default condor site {site_name!r}"
        effective = "condor"

    if to_write:
        write_sites(path, to_write, base_doc=doc)
    return action, effective


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("--style", choices=STYLES, required=True,
                        help="how the compute site is submitted to: a local "
                             "HTCondor pool, or Slurm through glite")
    parser.add_argument("--site", default="compute",
                        help="compute site name (default: compute, the "
                             "hosted catalogs' convention)")
    parser.add_argument("--full", action="store_true", default=None,
                        help="write a complete compute site rather than an "
                             "overlay (default: complete unless a hosted "
                             "catalog is named in ~/.pegasusrc)")
    parser.add_argument("--scratch", metavar="DIR",
                        help="full slurm site: shared scratch the workers and "
                             "submit host both see (default: $PWD/work)")
    parser.add_argument("--storage", metavar="DIR",
                        help="full slurm site: the site's output storage "
                             "(default: $PWD/storage)")
    parser.add_argument("--queue", metavar="Q",
                        help="partition/queue jobs submit to (pegasus.queue)")
    parser.add_argument("--project", metavar="ACCOUNT",
                        help="allocation charged (pegasus.project; "
                             "--account on Slurm)")
    parser.add_argument("--profile", action="append", default=[],
                        type=parse_profile, metavar="NS:KEY=VALUE",
                        help="extra profile on the site, e.g. "
                             "pegasus:glite.arguments=--constraint=avx512 or "
                             "env:PYTHONUNBUFFERED=1; repeatable")
    parser.add_argument("--train", action="append", default=[],
                        type=parse_profile, metavar="NS:KEY=VALUE",
                        help=f"profile on the '{TRAIN_TAG}' tag (LSTM "
                             "training only), e.g. pegasus:queue=long or "
                             "pegasus:runtime=21600; repeatable")
    parser.add_argument("-o", "--output", default="sites.yml",
                        help="where to write (default: sites.yml — the "
                             "name Pegasus picks up from the working "
                             "directory). Other sites in it are kept.")
    args = parser.parse_args()

    try:
        action, _ = ensure_sites_yml(
            args.output, args.site,
            wf_dir=os.path.dirname(os.path.abspath(__file__)),
            style=args.style, full=args.full, queue=args.queue,
            project=args.project, scratch=args.scratch, storage=args.storage,
            profiles=args.profile, train=args.train)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{args.output}: {action}")


if __name__ == "__main__":
    main()
