#set page(
  paper: "us-letter",
  margin: (x: 0.55in, y: 0.55in),
)

#set text(
  font: "Helvetica",
  size: 10pt,
  fill: rgb("#000000"),
)

#let garnet = rgb("#73000A")
#let black = rgb("#000000")
#let neutral = rgb("#5C5C5C")

#show heading.where(level: 1): set text(size: 18pt, weight: "bold", fill: garnet)
#show heading.where(level: 2): set text(size: 12pt, weight: "bold", fill: garnet)

= WavePart MATLAB and Python Comparison

This report compares representative WavePart outputs generated from the bundled sample dataset at indices 1 and 287. Each section shows the MATLAB and Python renderings of the same partition result in Cartesian and polar form.

#pagebreak()

== Case 1

#grid(
  columns: (1fr, 1fr),
  gutter: 0.18in,
  [
    #figure(
      image("assets/matlab_case_001_surface.png", width: 100%),
      caption: [MATLAB surface view for index 1.],
    )
  ],
  [
    #figure(
      image("assets/python_case_001_surface.png", width: 100%),
      caption: [Python surface view for index 1.],
    )
  ],
  [
    #figure(
      image("assets/matlab_case_001_polar.png", width: 100%),
      caption: [MATLAB polar view for index 1.],
    )
  ],
  [
    #figure(
      image("assets/python_case_001_polar.png", width: 100%),
      caption: [Python polar view for index 1.],
    )
  ],
)

#pagebreak()

== Case 287

#grid(
  columns: (1fr, 1fr),
  gutter: 0.18in,
  [
    #figure(
      image("assets/matlab_case_287_surface.png", width: 100%),
      caption: [MATLAB surface view for index 287.],
    )
  ],
  [
    #figure(
      image("assets/python_case_287_surface.png", width: 100%),
      caption: [Python surface view for index 287.],
    )
  ],
  [
    #figure(
      image("assets/matlab_case_287_polar.png", width: 100%),
      caption: [MATLAB polar view for index 287.],
    )
  ],
  [
    #figure(
      image("assets/python_case_287_polar.png", width: 100%),
      caption: [Python polar view for index 287.],
    )
  ],
)

#v(0.4em)

#set text(size: 9pt, fill: neutral)
Generated from `/Users/user/WavePart/data/wavespec2d_ex.mat` using local MATLAB R2025a and the Python `wavepart` package in this repository.
