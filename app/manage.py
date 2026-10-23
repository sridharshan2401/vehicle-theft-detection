"""Registry and review commands.

    python -m app.manage sessions
    python -m app.manage events --session <id>
    python -m app.manage associations [--session <id>]
    python -m app.manage review-association <id> confirm|reject [--note "..."]
    python -m app.manage add-vehicle --name "Blue hatchback" [--plate KA01AB1234] [--zone A1]
    python -m app.manage add-person --label "Owner" --images a.jpg b.jpg
    python -m app.manage add-person --label "Owner" --session <id> --profile P002
    python -m app.manage authorize --vehicle 1 --person 1
    python -m app.manage revoke --vehicle 1 --person 1
    python -m app.manage registry
    python -m app.manage purge --older-than-days 30 --yes

Authorisation is only ever granted here, by a person. Confirming an observed
association records a review decision; it does not authorise anyone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app.config import PROJECT_ROOT, load_config, resolve_path
from app.evidence import purge_old_sessions, safe_name
from app.storage import Storage, StorageError

REGISTRY_DIR = PROJECT_ROOT / "data" / "registry"


def _table(rows: list[dict], cols: list[str]) -> None:
    if not rows:
        print("(none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))[:60]) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, ""))[:60].ljust(widths[c]) for c in cols))


def _embeddings_from_images(images: list[str]) -> tuple[np.ndarray, str]:
    from app.hardware import inspect_hardware, select_device
    from app.person_reid import build_reid_backend

    cfg = load_config()
    device, _ = select_device(cfg.device.device, "off", inspect_hardware())
    backend, status = build_reid_backend(cfg.reid, device)
    if backend is None:
        raise SystemExit(f"ReID is unavailable: {status}")
    if not backend.person_specific:
        print(f"warning: '{backend.name}' is not a person ReID model; authorisation matching stays off "
              "unless reid.allow_generic_backend_for_authorization is true.")
    crops = []
    for p in images:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read image {p}")
        crops.append(img)
    return backend.embed(crops, [None] * len(crops)), backend.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.manage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sessions")
    p = sub.add_parser("events")
    p.add_argument("--session", required=True)
    p = sub.add_parser("associations")
    p.add_argument("--session")
    p = sub.add_parser("review-association")
    p.add_argument("association_id", type=int)
    p.add_argument("decision", choices=["confirm", "reject"])
    p.add_argument("--note")
    p = sub.add_parser("add-vehicle")
    p.add_argument("--name", required=True)
    p.add_argument("--plate")
    p.add_argument("--zone", help="parking zone name this vehicle normally uses")
    p.add_argument("--notes")
    p = sub.add_parser("add-person")
    p.add_argument("--label", required=True)
    p.add_argument("--images", nargs="+", help="cropped full-body photos of the person")
    p.add_argument("--session", help="take the appearance profile from a past session")
    p.add_argument("--profile", help="profile id in that session, e.g. P002")
    p.add_argument("--notes")
    for name in ("authorize", "revoke"):
        p = sub.add_parser(name)
        p.add_argument("--vehicle", type=int, required=True)
        p.add_argument("--person", type=int, required=True)
    sub.add_parser("registry")
    p = sub.add_parser("purge")
    p.add_argument("--older-than-days", type=int, required=True)
    p.add_argument("--yes", action="store_true", help="actually delete")
    args = ap.parse_args(argv)

    cfg = load_config()
    db = Storage(resolve_path(cfg.storage.db_path))
    try:
        if args.cmd == "sessions":
            _table(db.list_sessions(), ["session_id", "mode", "started_at", "ended_at", "video_path"])
        elif args.cmd == "events":
            _table(db.list_events(args.session), ["frame", "video_ts", "risk_level", "category", "vehicle_id",
                                                  "person_track_id", "message"])
        elif args.cmd == "associations":
            _table(db.list_associations(args.session), ["association_id", "session_id", "person_track_id", "vehicle_id",
                                                        "status", "confidence", "review_status"])
        elif args.cmd == "review-association":
            db.review_association(args.association_id, "confirmed" if args.decision == "confirm" else "rejected", args.note)
            print(f"Association {args.association_id} marked {args.decision}ed (this does not authorise anyone).")
        elif args.cmd == "add-vehicle":
            rid = db.register_vehicle(args.name, args.plate, args.zone, args.notes)
            print(f"Registered vehicle #{rid}")
        elif args.cmd == "add-person":
            REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
            if args.images:
                embs, backend = _embeddings_from_images(args.images)
            elif args.session and args.profile:
                root = resolve_path(cfg.evidence.output_dir) / safe_name(args.session)
                files = sorted(root.glob("profiles_*.npz"))
                if not files:
                    raise SystemExit(f"No saved profiles in {root}")
                data = np.load(files[0])
                if args.profile not in data:
                    raise SystemExit(f"Profile {args.profile} not in {files[0].name}; have {list(data.keys())}")
                embs = data[args.profile].reshape(1, -1)
                backend = files[0].stem.split("_", 1)[1]
            else:
                raise SystemExit("Give --images, or --session and --profile")
            path = REGISTRY_DIR / f"{safe_name(args.label)}_{backend}.npy"
            np.save(path, embs.astype(np.float32))
            rid = db.register_person(args.label, str(path), backend, args.notes)
            print(f"Registered person #{rid} ({backend} embedding saved to {path})")
        elif args.cmd == "authorize":
            db.authorize(args.vehicle, args.person)
            print(f"Person #{args.person} is now authorised for vehicle #{args.vehicle}")
        elif args.cmd == "revoke":
            db.revoke(args.vehicle, args.person)
            print(f"Authorisation removed for person #{args.person} on vehicle #{args.vehicle}")
        elif args.cmd == "registry":
            print("Vehicles:")
            _table(db.list_vehicles(), ["vehicle_record_id", "name", "registration_number", "parking_zone"])
            print("\nAuthorised people:")
            _table(db.list_people(), ["person_record_id", "label", "reid_backend", "embedding_path"])
            print("\nAuthorisations:")
            for v in db.list_vehicles():
                people = db.authorized_people_for_vehicle(v["vehicle_record_id"])
                print(f"  vehicle #{v['vehicle_record_id']}: {[p['label'] for p in people] or 'nobody'}")
        elif args.cmd == "purge":
            if not args.yes:
                print("Dry run. Add --yes to delete session folders older than "
                      f"{args.older_than_days} days from {resolve_path(cfg.evidence.output_dir)}.")
                return 0
            removed = purge_old_sessions(resolve_path(cfg.evidence.output_dir), args.older_than_days, db)
            print(json.dumps({"removed_sessions": removed}, indent=2))
    except StorageError as exc:
        print(f"error: {exc}")
        return 2
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
