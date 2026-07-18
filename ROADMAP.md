# Roadmap

Planned for the next minor release (v0.2.0):

- **Dimension weights that take effect.** `Sampler.set_dimension_weights`
  currently stores the weights without applying them; the next release
  wires them into the optimisation, with tests and updated documentation.
- **User-supplied candidate lists.** A `ParameterSpace.from_candidates`
  constructor that builds the space from an existing list of feasible
  points (a mesh from another program, the valid settings of a control
  system), with designs selected from that list. Exclusion zones and
  focus regions apply to the list by coordinates.

Suggestions and bug reports are welcome in the
[issue tracker](https://github.com/acanbay/mergen/issues).
