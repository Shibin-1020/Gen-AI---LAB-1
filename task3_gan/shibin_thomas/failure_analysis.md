<!-- AUTO-DRAFT: template. evaluate_local.py replaces this file with the automatically ranked visual failure candidates of the full run. Look at the grids, describe each failure in your own words, then delete this line. -->
# Task 3 - Visual failure analysis

_Pending the full training run._ After `run_pipeline.py` finishes, this file lists, for each direction, the images
with the highest cycle-reconstruction error, the lowest content preservation (Inception cosine input vs
translation) and the least realistic outputs (farthest from any real target image). Each comes with an
input | translation | reconstruction grid. These are then described by hand: type of failure (colour shift,
texture artifacts, lost structure, mode collapse, checkerboard, over-/under-stylisation), likely cause and a
testable fix.
