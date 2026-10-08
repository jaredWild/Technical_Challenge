# Visual Memory POC

This is a proof of concept for a low-resource camera device that observes its environment, identifies meaningful visual changes, creates persistent memories, and allows those memories to be retrieved later.

## What I Built

The system uses a webcam as the device's visual input.

Rather than analyzing every camera frame with an external vision model, it first performs inexpensive local change detection. Frames that appear sufficiently different are compressed and queued for semantic analysis using Gemini.

A stored memory contains:

- Timestamp
- Important visible entities
- Environment / scene description
- Meaningful change since the previous memory
- Importance score from 1–10
- The compressed source image

Memories are stored persistently in a SQLite database.

A separate query script allows stored memories to be searched and reviewed by:

- Search term
- Most recent memories
- Minimum importance
- Last appearance of an entity or term
- Memory ID, including viewing the associated stored image

## How It Works

The webcam is sampled at 5 FPS.

Each sampled frame is resized to 25% of its original dimensions for inexpensive local comparison. The absolute pixel difference between consecutive sampled frames is calculated, and frames exceeding a configurable threshold are selected as potential memories.

The first frame always establishes the initial memory.

Because gradual changes may never exceed the frame-to-frame threshold, the system also periodically compares the current scene against the last successfully stored memory. This provides a second opportunity to detect accumulated changes that occurred slowly.

Selected frames are JPEG-compressed before being placed in a bounded queue. A background worker processes this queue separately from the camera loop so network/API latency does not interrupt camera capture.

For memories after the initial memory, Gemini receives both the previous remembered image and the new image. It returns structured information describing the current scene and what meaningfully changed between the two.

Successfully analyzed memories are then stored in SQLite together with their compressed image.

## Running the Project

### 1. Install dependencies

```bash
pip install opencv-python numpy python-dotenv google-genai pydantic
```

Alternatively, if using the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configure the Gemini API key

Create a `.env` file in the project directory:

```text
GEMINI_API_KEY=your_api_key_here
```

The project currently uses `gemini-3.5-flash-lite`.

### 3. Run the camera

```bash
python camera.py
```

Press `q` to stop camera capture.

After capture stops, any frames remaining in the analysis queue are processed before the program exits.

### 4. Query stored memories

```bash
python query_memories.py
```

The CLI provides several retrieval options, including search, recent memories, important memories, last-seen queries, and viewing a specific memory and its stored image.

`check_memories.py` can also be used to inspect stored memories directly.

## Important Assumptions and Decisions

### Local filtering before AI analysis

Sending every camera frame to a vision API would be slow, expensive, and inappropriate for the target resource constraints.

The system therefore performs simple change detection locally and only sends selected observations for semantic analysis.

### Limited local compute

The local workload consists mainly of:

- Webcam capture
- Resizing
- Pixel-difference calculations
- JPEG encoding
- SQLite access

The expensive semantic image understanding is performed remotely, taking advantage of the challenge's assumption that network access is available.

### Memory definition

A memory represents a meaningful change in the observed environment rather than every captured frame.

The first observation establishes the initial state. Later memories represent changes relative to the previous successfully analyzed memory.

This keeps storage requirements lower while making stored observations more useful for later retrieval.

### Storage

Images are JPEG-compressed before storage.

Both semantic metadata and the associated JPEG are stored together in SQLite. This keeps the POC self-contained and avoids maintaining synchronization between a database and a separate image directory.

### Background processing

Vision API calls are slower than camera sampling, so analysis runs in a separate worker thread.

Selected frames are placed into a bounded queue. This allows the camera to continue operating while analysis catches up during periods of lower activity.

The bounded queue prevents uncontrolled RAM growth if observations are produced faster than the external model can process them.

### Model choice

A lightweight Gemini model was selected because this is a proof of concept and the required task is relatively focused.

The model/API layer could be replaced without substantially changing the rest of the architecture.

## Current Limitations

Change detection currently uses mean absolute pixel difference. This is inexpensive and effective enough for the proof of concept, but it can be sensitive to camera motion, lighting changes, and the selected threshold.

The appropriate threshold is also context-dependent. A stationary indoor camera and a moving camera would likely require different tuning or different change-detection approaches.

If the analysis queue reaches its maximum size, newly selected observations are currently dropped.

The importance score is model-generated and would need evaluation across a more varied dataset. My testing environment mostly consisted of routine movement within a single room, so naturally most observed changes received relatively low importance scores.

The system also depends on network access and availability of the external vision API.

## What I Would Do Next

With more development time, I would focus on:

- Replace simple absolute pixel difference with more robust scene-change or motion detection, particularly for moving cameras.
- Make change thresholds adaptive to the device and environment rather than manually configured.
- Improve queue overflow behavior so potentially valuable observations are prioritized instead of automatically dropping the newest frame.
- Add semantic deduplication so visually different frames that represent the same state do not create unnecessary memories.
- Evaluate and calibrate the importance scoring system using more varied real-world scenarios.
- Add storage-retention policies so old or low-value memories can be summarized or removed as storage approaches its limit.
- Add resilience around network outages, API rate limits, or other failures.
