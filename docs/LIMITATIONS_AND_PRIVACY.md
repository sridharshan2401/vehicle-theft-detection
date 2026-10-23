# Limitations and privacy

## What the system cannot do

* **It cannot tell you who is a thief.** It reports observations (someone near a car,
  someone who seemed to get in, a car that moved) with confidences. A HIGH alert means
  "look at this", nothing more.
* **Exit and entry are inferred from track start and end.** Someone walking behind a car
  and out of sight looks like an entry. Someone who was hidden behind a car and steps out
  can look like an exit (there is a penalty for this, but it is a heuristic).
* **Image space is not the real world.** "Near the car" is measured in pixels and box
  sizes. Perspective makes far objects small; a top-down camera and a side camera put the
  doors in very different places in the image. The door-zone test is a rough approximation.
* **Associations are not ownership.** "P003 got out of V001" is something the camera
  appeared to show. It never makes P003 authorised. Authorisation only comes from the
  registry, entered by a person.
* **ReID is a guess.** The default backend on the dev machine is a generic ImageNet
  network, not a person-ReID model, so reconnections are rare and conservative.
  Even with OSNet, clothing changes, lighting and camera angle break matches. Uncertain
  cases are shown as `?P002` or "new (near P002?)", never silently merged.
* **Clothing is coarse.** Thirteen colour names, a few garment types (when the SegFormer
  model is installed), and a plain/patterned hint. No brands or exact designs. Low light
  and small people produce low-quality results, which are marked as such.
* **No number plates.** A registered vehicle is linked through the parking zone it uses.
  A different car parked in that spot would be treated as the registered one.
* **One camera, one video at a time.** No cross-camera tracking.
* **Responsive mode skips frames** when the GPU cannot keep up. Every skip is counted
  and shown, and the tracker is reset after long gaps, but events can be missed. Use
  complete mode for anything you might rely on.
* **Not evaluated on staged footage.** See docs/TESTING.md for what is and is not measured.

## Privacy and responsible use

Video of people in a car park is personal data in most places (e.g. under GDPR or
India's DPDP Act). Before using this on real footage, check that you are allowed to
record and analyse it and for what purpose.

What the software does by default:

* **Stays local.** No network calls during analysis. Models are downloaded once by an
  explicit script from fixed official sources.
* **Never modifies the source video.** Annotated output is a separate file.
* **No face recognition.** `privacy.face_processing` is `false` and there is no face
  model in the project. If you add one, make it opt-in and get consent.
* **Appearance embeddings are optional.** `privacy.save_profile_embeddings: false` stops
  `profiles_*.npz` from being written. Registry embeddings in `data/registry/` exist only
  for people you register on purpose.
* **Retention.** `evidence.retention_days` deletes old session folders at start-up (only
  folders containing the app's `session.json` marker). Or run
  `python -m app.manage purge --older-than-days 30 --yes`.
* **Careful wording.** Alerts say "unrecognised person", "appeared to get in", "possible
  unauthorised vehicle movement". Appearance (clothing, colour) never raises risk on its own.
* **Logs** contain track ids and file paths, not names, unless you register a person
  with a name as the label.
* **Git**: `.gitignore` excludes videos, evidence, the database, embeddings and model weights.

Recommended practice:

* Give evidence folders the same access controls as the original CCTV.
* Have a person review every MEDIUM+ alert before acting on it; record the decision
  (`app.manage review-association` exists for associations).
* Do not use clothing or appearance attributes to search for people outside the
  incident being investigated.
