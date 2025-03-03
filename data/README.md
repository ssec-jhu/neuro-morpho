# The Data

## Raw

Raw data is stored in the `raw` directory and has the following naming 
convention:

**Input files:**

```Realistic-SBR-<n>-Sample-<m>-time-<t>.pgm```

Where `<n>` indicates the quality of the image wrt to noise, `<m>` is
an id that differentiates between different
dendrites, and `<t>` indicates how much time the sample has ....

- `<n>` can range from `1` to `5`, where `1` is the most noisy and
  `5` is the most clear.

- `<m>` can range from `1` to `100` and is the id for each individual
  dendrite

- `<t>` seems to be associated with time and only has the value `100.00`

**Output Files:**

``<type>-Sample-<m>-time-<t>.pgm``

Where `<type>` indicates what kind of output and `<m>`/`<t>` mean the 
same as above and relate the output files to the input files.

- `<type>` can be one two options:
  - `Segmented` this is the segmentation map assocaited with the input
  - `Skeleton` this is a single pixel line traversing the dendrite

**Misc Files:**

``SWC-Sample-<m>-time-<t>.swc``

Where `<m>`/`<t>` mean the same as above and relate the input/output files
to the these files. SWC files are text files that represent the skeleton
of the dendrite but because of the format contain additional information
that help derive branch and tip locations.
