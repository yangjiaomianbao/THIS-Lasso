"""Finalize new audit inventories; never write scientific or original source files."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/refactor"
OUTPUT_NAMES = [
    "S6_change_overview.csv", "S6_changes_summary.json", "S6_link_report.csv",
    "S6_link_summary.json", "S6_space_summary.json", "S6_delivery_manifest.csv",
    "S6_delivery_checks.json",
]


def sha(path):
    p = ROOT / path
    if p.is_symlink():
        return hashlib.sha256(os.fsencode(os.readlink(p))).hexdigest()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(name):
    with (OUT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def tree():
    rows = {}
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        for name in files + [d for d in dirs if (Path(base) / d).is_symlink()]:
            p = Path(base) / name
            s = p.lstat()
            assert stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode), p
            rows[p.relative_to(ROOT).as_posix()] = s
    return rows


def category(path):
    if path.startswith("archive/"):
        return "archive"
    if path.startswith(("src/", "scripts/")):
        return "code"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("configs/"):
        return "configs"
    return "docs"


def csv_bytes(rows, fields):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fields)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def measured():
    entries = tree()
    seen = set()
    blocks = 0
    for s in entries.values():
        key = (s.st_dev, s.st_ino)
        if key not in seen:
            blocks += s.st_blocks * 512
            seen.add(key)
    return dict(
        entries=len(entries),
        regular_file_bytes=sum(s.st_size for s in entries.values() if stat.S_ISREG(s.st_mode)),
        symlink_text_bytes=sum(s.st_size for s in entries.values() if stat.S_ISLNK(s.st_mode)),
        unique_inode_allocated_bytes=blocks,
    )


def main():
    for name in OUTPUT_NAMES:
        assert not (OUT / name).exists(), f"Refusing to overwrite: {name}"
    baseline = {r["path"]: r for r in read_csv("S1_baseline_manifest.csv")}
    original = {r["path"] for r in read_csv("S0_inventory.csv")}
    deleted = {r["path"]: r for r in read_csv("S5_deletion_log.csv")}
    s2 = {r["path"]: r for r in read_csv("S2_changed_files.csv")}
    current = tree()
    assert original - current.keys() == deleted.keys()
    modified = {p for p in original & current.keys() if sha(p) != baseline[p]["sha256"]}
    assert modified == {p for p, r in s2.items() if r["action"] == "modified"} | {"README.md"}
    for p, r in s2.items():
        assert sha(p) == r["after_sha256"], p
    assert sha("archive/snapshots/README_before_S3.md") == baseline["README.md"]["sha256"]
    planned = {"docs/refactor/" + name for name in OUTPUT_NAMES}
    additions = (current.keys() | planned) - original
    assert all(p.startswith(("docs/", "archive/", "src/", "tests/")) for p in additions)
    changes = []
    for action, paths in [("modified", modified), ("added", additions), ("deleted", deleted.keys())]:
        for p in sorted(paths):
            cat = "deleted" if action == "deleted" else category(p)
            reason = (
                "S5 exact-path optional cache cleanup; verified member in archive/caches/S5_python_cache_backup.tar.gz"
                if action == "deleted" else
                "S2 shared implementation extraction or compatibility export" if cat == "code" else
                "S2 additional mathematical regression checks" if cat == "tests" else
                "History navigation, original README snapshot or verified cache recovery backup" if cat == "archive" else
                "Audit evidence, navigation, reproducibility guide or README refresh"
            )
            changes.append(dict(path=p, category=cat, action=action, reason=reason,
                                before_sha256=baseline[p]["sha256"] if p in original else ""))
    (OUT / "S6_change_overview.csv").write_bytes(csv_bytes(changes, list(changes[0])))
    counts = {}
    for cat in ["code", "configs", "tests", "docs", "archive", "deleted"]:
        counts[cat] = {a: sum(r["category"] == cat and r["action"] == a for r in changes)
                       for a in ["modified", "added", "deleted"]}
    assert counts["code"] == dict(modified=6, added=8, deleted=0)
    assert counts["tests"] == dict(modified=0, added=1, deleted=0)
    assert not any(counts["configs"].values())
    write_json(OUT / "S6_changes_summary.json", dict(
        status="PASS", original_path_inventory="S0_inventory.csv",
        original_content_reference="S1_baseline_manifest.csv",
        original_entries=len(original), planned_final_inventory=True,
        counts=counts, renamed_files=0, moved_files=0,
        production_code_changes_after_S2=0,
        note="Audit scripts and trial outputs under docs/refactor are audit deliverables, classified as docs; no new scientific experiments.",
    ))
    subprocess.run([sys.executable, "-B", str(OUT / "S3_link_audit.py"), "S6"], cwd=ROOT, check=True)
    link = json.loads((OUT / "S6_link_summary.json").read_text())
    assert not link["unresolved"]
    run = json.loads((OUT / "S6_work/final_20261020_verified/result.json").read_text())
    assert run["status"] == "PASS"
    formal = read_csv("S1_R5_formal_manifest.csv")
    for r in formal:
        assert sha(r["path"]) == r["sha256"], r["path"]
    for name in ["S1_audit_artifacts.csv", "S2_artifact_manifest.csv"]:
        for r in read_csv(name):
            assert sha(r["path"]) == r["sha256"], r["path"]
    checks = dict(status="PASS", scientific_audit="S6_work/final_20261020_verified/result.json",
                  baseline_changes_limited_to_S2_README_and_approved_caches=True,
                  original_README_snapshot_exact=True, S1_S2_audit_artifacts_unchanged=True,
                  formal_R5_files_exact=len(formal), original_missing_exactly_approved_caches=len(deleted),
                  current_local_links_unresolved=0, current_local_links_scanned=link["local_links"],
                  delivery_manifest_excludes_only_itself=True,
                  delivery_manifest_rows_verified=True)
    write_json(OUT / "S6_delivery_checks.json", checks)
    before = json.loads((OUT / "S3_preflight.json").read_text())
    space_path = OUT / "S6_space_summary.json"
    manifest_path = OUT / "S6_delivery_manifest.csv"
    space = dict(
        status="PASS", baseline="S3_preflight.json before S3-S6 additions and cleanup",
        regular_file_bytes_before=before["logical_file_bytes_before"],
        unique_inode_allocated_bytes_before=before["project_unique_allocated_bytes_before"],
        final={}, regular_file_bytes_net_reduction=0, logical_net_reduction_MiB=0.0,
        unique_inode_allocated_bytes_net_reduction=0, allocated_net_reduction_MiB=0.0,
        includes="All current files, symlink text accounting and final manifest; optional cache backup and all audit outputs retained.",
        excludes="Directory inode metadata, filesystem snapshots, compression/reflink effects and external files; allocated byte delta is not a claim about global disk free space.",
        manifest="S6_delivery_manifest.csv excludes itself to avoid self-referential SHA256",
    )
    write_json(space_path, space)
    records = []
    for p, s in sorted(tree().items()):
        records.append(dict(path=p, type="symlink" if stat.S_ISLNK(s.st_mode) else "file",
                            size=s.st_size, sha256=sha(p),
                            category=category(p) if p in modified or p in additions else "unchanged"))
    fields = list(records[0])
    # Only these new audit outputs are rewritten while their size accounting converges.
    # Hashes have fixed width, so JSON/CSV lengths stabilize without a self-hash.
    for _ in range(10):
        for r in records:
            if r["path"] == "docs/refactor/S6_space_summary.json":
                r.update(size=space_path.stat().st_size, sha256=sha(r["path"]))
        manifest_path.write_bytes(csv_bytes(records, fields))
        final = measured()
        new = dict(space)
        new["final"] = final
        new["regular_file_bytes_net_reduction"] = space["regular_file_bytes_before"] - final["regular_file_bytes"]
        new["logical_net_reduction_MiB"] = new["regular_file_bytes_net_reduction"] / 1048576
        new["unique_inode_allocated_bytes_net_reduction"] = space["unique_inode_allocated_bytes_before"] - final["unique_inode_allocated_bytes"]
        new["allocated_net_reduction_MiB"] = new["unique_inode_allocated_bytes_net_reduction"] / 1048576
        if new == space:
            break
        space = new
        write_json(space_path, space)
    else:
        raise AssertionError("Space accounting failed to converge")
    actual = tree()
    assert actual.keys() - {"docs/refactor/S6_delivery_manifest.csv"} == {r["path"] for r in records}
    for r in records:
        assert actual[r["path"]].st_size == r["size"] and sha(r["path"]) == r["sha256"], r["path"]
    assert measured() == space["final"]
    assert {r["path"] for r in changes if r["action"] == "added"} == actual.keys() - original
    print(json.dumps(dict(status="PASS", counts=counts, delivery_rows=len(records),
                          manifest_sha256=sha("docs/refactor/S6_delivery_manifest.csv"),
                          space=space), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
