# Test stormpy using gridworld problem from POMDPModels.jl
import stormpy

tra_file = "grid_world.tra"
lab_file = "grid_world.lab"

model = stormpy.build_sparse_model_from_explicit(tra_file, lab_file)
# property = "Pmax=? [ ! \"bad\" U<=30 \"good\"]"
property = "Pmax=? [ G ! \"bad\"]"
# property = "Pmax=? [F<=30 \"good\"]"
# property = "Pmin=? [! \"bad\" U \"good\"]"

properties = stormpy.parse_properties(property)
result = stormpy.model_checking(model, properties[0], extract_scheduler=True)
assert result.has_scheduler
