"""Dense native shared boundaries must retain distinct nearby vertices."""
import math

import numpy as np
import pytest


def test_dense_source_rim_retains_sub_weld_edges(tmp_path):
    meshio=pytest.importorskip("meshio")
    from hornlab_mesher.mesher import _postprocess_mesh
    from hornlab_mesher.normals import open_shell_wall_orientation_references
    count=2048
    az=np.arange(count)*2*math.pi/count
    throat=np.column_stack((np.cos(az),np.sin(az),np.zeros(count)))
    mouth=np.column_stack((1.01*np.cos(az),1.01*np.sin(az),np.ones(count)))
    points=np.vstack((throat,mouth,[[0,0,0]]))
    i=np.arange(count);nxt=(i+1)%count
    wall=np.vstack((np.column_stack((i,i+count,nxt)),np.column_stack((nxt,i+count,nxt+count))))
    source=np.column_stack((np.full(count,2*count),i,nxt))
    triangles=np.vstack((wall,source))
    tags=np.r_[np.ones(2*count,dtype=int),np.full(count,2,dtype=int)]
    raw=tmp_path/"raw.msh";out=tmp_path/"native.msh"
    meshio.write(raw,meshio.Mesh(points,[("triangle",triangles)],cell_data={"gmsh:physical":[tags],"gmsh:geometrical":[tags]},
        field_data={"SD1G0":np.array([1,2]),"SD1D1001":np.array([2,2])}),file_format="gmsh22",binary=False)
    refs,ns=open_shell_wall_orientation_references(np.stack((throat,mouth),axis=1),closed=True)
    assert np.linalg.norm(throat[1]-throat[0])<.005
    info=_postprocess_mesh(raw,out,"z",False,open_shell_wall_points_mm=refs,open_shell_wall_normals=ns,weld_near_duplicates=False)
    result=meshio.read(out)
    assert info.n_triangles==3*count
    assert len(result.points)==2*count+1
    cells=result.cells_dict["triangle"]
    groups=result.cell_data_dict["gmsh:physical"]["triangle"]
    src=cells[groups==2]
    assert len(src)==count
    tri=result.points[src]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    assert np.all(normals[:,2]>0)
    area=np.sum(np.linalg.norm(normals,axis=1))/2
    assert area==pytest.approx(math.pi,rel=2e-6)


def test_actual_native_build_never_calls_approximate_welder(monkeypatch,tmp_path):
    pytest.importorskip("gmsh")
    meshio=pytest.importorskip("meshio")
    import hornlab_mesher.mesher as mesher
    from hornlab_mesher.config_builder import build_from_config
    def forbidden(*args,**kwargs):raise AssertionError("native shared edges must not use approximate welding")
    monkeypatch.setattr(mesher,"_weld_near_duplicate_vertices",forbidden)
    c={"formula":"OSSE-AXIAL","axial_scale":1,
        "profile":{"L_mm":1,"r0_mm":1,"a_deg":5,"a0_deg":0,"k":1,"s":0},
        "mesh":{"throat_res_mm":.01,"mouth_res_mm":.05,"scale_to_metres":False,"allow_large_mesh":True},"source":{"source_shape":0}}
    result=build_from_config(c,tmp_path/"small-native.msh")
    assert result.n_triangles>0
    assert result.physical_groups=={1:"SD1G0",2:"SD1D1001"}
    output=meshio.read(tmp_path/"small-native.msh")
    cells=output.cells_dict["triangle"]
    groups=output.cell_data_dict["gmsh:physical"]["triangle"]
    source=cells[groups==2];wall=cells[groups==1]
    tri=output.points[source]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    assert np.all(cross[:,2]>0) and np.max(abs(tri[:,:,2]))==0
    assert np.sum(np.linalg.norm(cross,axis=1))/2==pytest.approx(math.pi,rel=.001)
    edges=np.sort(np.concatenate((source[:,[0,1]],source[:,[1,2]],source[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    rim=unique[counts==1]
    wall_edges=np.sort(np.concatenate((wall[:,[0,1]],wall[:,[1,2]],wall[:,[2,0]])),axis=1)
    wall_set={tuple(edge) for edge in wall_edges}
    assert len(rim)>500 and all(tuple(edge) in wall_set for edge in rim)
