import json
import pytest
from ffgd.problem_io import load_problem


@pytest.mark.parametrize('location',['root','task','scene','obstacle'])
def test_hidden_solution_fields_rejected_before_robot_construction(tmp_path,location):
    problem=dict(initial={},task=dict(duration=4.,poses=[]),scene=dict(boxes=[]))
    if location=='root':problem['terminal_q17']=[0]*17
    elif location=='task':problem['task']['terminal_q17']=[0]*17
    elif location=='scene':problem['scene']['solution_path']='hidden.npy'
    else:problem['scene']['boxes']=[dict(center=[1,2,3],half_size=[.1]*3,certificate='hidden.npz')]
    path=tmp_path/'problem.json';path.write_text(json.dumps(problem))
    with pytest.raises(ValueError,match='unknown'):load_problem(path)
