# Smart Image Quality Enhancement and Object Segmentation

A Streamlit app that takes a **low-light object image**, removes noise, enhances contrast, segments the main object and shows every stage.

## Why low-light object images?
A dim photo has two real problems: the pixels are bunched near black (low contrast) and the sensor noise is strong. That makes every required concept useful rather than decorative: the histogram shows the problem, CLAHE fixes contrast, blur removes noise, thresholding separates the object, morphology cleans the mask, and connected components pick the object.

## Project structure
```
smart-image-segmentation/
├── app.py                  # the Streamlit application
├── requirements.txt
├── README.md
├── REPORT.md               # content for the college report
├── flowchart.png           # workflow diagram
├── make_sample_image.py    # creates the sample image
├── make_flowchart.py       # creates flowchart.png
└── sample_images/
    └── sample_low_light.jpg   # synthetic demo image (replace with real photos)
```

## Run on Windows (Command Prompt)
```
1.  mkdir smart-image-segmentation
    cd smart-image-segmentation
2.  python -m venv venv
3.  venv\Scripts\activate
4.  pip install -r requirements.txt
5.  (copy app.py, requirements.txt and sample_images into this folder)
    python make_sample_image.py      (only if the sample image is missing)
6.  streamlit run app.py
7.  Open http://localhost:8501 (it opens automatically)
8.  Upload a JPG/PNG in the sidebar, or click "Use sample low-light image"
9.  Click "Process Image"
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first.

## Pipeline
Input → Grayscale → Noise removal/smoothing → Histogram → CLAHE/Equalization → Global / Otsu / Adaptive threshold → Morphology → Connected components (largest region, 8-connectivity) → Final object

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` (or streamlit/matplotlib) | Activate the venv, then `pip install -r requirements.txt`. |
| OpenCV install fails | `python -m pip install --upgrade pip`, then `pip install opencv-python`. Use 64-bit Python 3.9-3.12. |
| `'streamlit' is not recognized` | Venv not active, or use `python -m streamlit run app.py`. |
| Upload does nothing / error | Only JPG, JPEG, PNG are accepted. Corrupt files show an error message instead of crashing. |
| Mask is inverted (background white) | Set **Object type** manually to "Bright object on dark background" or "Dark object on bright background". |
| Thresholding gives an incorrect mask | Otsu usually works best. For Global, move the threshold slider (low-light images need low values, e.g. 30-60). |
| Object not detected ("No object detected") | Try Otsu, increase CLAHE clip limit, or reduce the morphology kernel (a big kernel can erase small objects). |
| Background detected instead of object | The object touches the border or is darker than the background: change **Object type**. |
| Adaptive threshold gives speckles | Adaptive compares pixels with local averages, so smooth object interiors are not filled. Use a larger block size or prefer Otsu for objects (adaptive suits documents). |

## Honest limitations
- The sample image is synthetic, so results on it prove the code works, not how well it performs in general.
- Single-object segmentation by intensity only: cluttered backgrounds, shadows or objects similar in brightness to the background will fail.
- No accuracy figure is claimed; the app reports only measured values (mean intensity, threshold, areas).

## Viva quick answers
- **Why blur before CLAHE?** Enhancement also amplifies noise, so reduce noise first.
- **CLAHE vs normal equalization?** CLAHE works on tiles with a clip limit, so contrast is local and noise is not over-boosted.
- **How does Otsu choose T?** It tries every T and picks the one that best separates the two histogram groups (maximum between-class variance).
- **Opening vs closing?** Opening = erosion then dilation (removes specks). Closing = dilation then erosion (fills gaps).
- **Why connected components?** After thresholding, leftover noise blobs remain; keeping the largest connected region isolates the object.

## Requirement checklist
| Requirement | Implemented? | Where |
|---|---|---|
| Histogram | Yes | Sections 3 and 4 |
| Histogram Equalization / CLAHE | Yes | Section 4 (selectable) |
| Noise Removal | Yes | Median blur, Section 2 |
| Smoothing | Yes | Gaussian blur, Section 2 |
| Thresholding | Yes | Global, Section 5 |
| Otsu Thresholding | Yes | Section 5 |
| Adaptive Thresholding | Yes | Section 5 |
| Morphological Operations | Yes | Opening, closing, dilation, erosion, Section 6 |
| Connectivity | Yes | Connected components, Section 6 |
| Original Output | Yes | Section 1 |
| Enhanced Output | Yes | Section 4 |
| Thresholded Output | Yes | Section 5 |
| Final Segmented Output | Yes | Section 7 |
| Streamlit Interface | Yes | `app.py` |
| Flowchart | Yes | `flowchart.png` |
| Project Report | Yes (text; add your own measured results) | `REPORT.md` |
