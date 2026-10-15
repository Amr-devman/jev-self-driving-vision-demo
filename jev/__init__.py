"""Jev Vision: object detection over a dashcam video with TypeSafe Jev.

The pipeline runs in four stages, and only the second one costs money:

    extract -> detect -> render -> encode

See the README for what each stage does and how the pieces fit together. The
command line entry point is jev_vision.py in the repository root, which wires
these modules together.
"""
