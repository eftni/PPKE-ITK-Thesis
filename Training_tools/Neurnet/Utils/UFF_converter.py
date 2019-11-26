import uff
import graphsurgeon as gs
import tensorflow as tf
import numpy as np

dyn_graph = gs.DynamicGraph("binary_alt.pb")
nodes = [n.name for n in dyn_graph.as_graph_def().node]
list = dyn_graph.find_nodes_by_name('Relu')
num = 0
for node in list:
    parent_list = dyn_graph.find_node_inputs_by_name(node, "Add")
    if len(parent_list) == 0: continue
    parent = parent_list[0]
    if (str(parent.input).find("Pad") != -1):
        gp = parent.input[0] if str(parent.input[0]).find("Pad") != -1 else parent.input[1]
        other = parent.input[1] if str(parent.input[0]).find("Pad") != -1 else parent.input[0]

        t_const = gs.create_node("Tconst" + str(num), 'Const', dtype=tf.float16, permutation=[2,1,0])
        dyn_graph.append(t_const)
        trans = gs.create_node('Transpose' + str(num), 'Transpose', inputs=[gp, t_const.name], dtype=tf.int8, permutation=np.array([2,1,0]))
        dyn_graph.append(trans)
        num = num + 1
        parent.input[:] = [other, trans.name]

uff_model = uff.from_tensorflow(dyn_graph.as_graph_def(),
                                output_nodes=["concatenate_1/concat", "concatenate_2/concat"],
                                output_filename="GS_model.uff", text=True)

'''uff.from_tensorflow_frozen_model("binary_alt.pb", output_nodes=["concatenate_1/concat", "concatenate_2/concat"],
                                 output_filename="batch_model.uff", text=False)'''
