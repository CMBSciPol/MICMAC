"""
Module to create templates for spatial variability
(spv stands for SPatial Variability)
"""
import numpy as np
import healpy as hp
import copy
from anytree import Node, RenderTree
import jax
import yaml
import jax.numpy as jnp
import chex as chx



#### Lower level functions
def read_spv_config(yaml_file_path):
    """Reads yaml file with info of spv configuration
    and creates a dictionary from there"""
    with open(yaml_file_path, 'r') as file:
        dict_params_spv = yaml.safe_load(file)

    return dict_params_spv


def build_tree_from_dict(node_dict, parent=None):
    """Recursive function to build a tree form a dict"""
    for key, value in node_dict.items():
        if isinstance(value, dict):
            node = Node(key, parent=parent)
            build_tree_from_dict(value, parent=node)
        else:
            Node(key, parent=parent, value=value)

    return


def count_betas_in_tree(node):
    count = 0
    if node.name.startswith('b'):
        count += 1
    for child in node.children:
        count += count_betas_in_tree(child)
    
    return count


def select_child_with_name(parent_node, child_name):
    """Return child with a given name"""
    for child in parent_node.children:
        if child.name == child_name:
            return child
    
    return None


def fill_betas(node):
    """Check if all the b params have values, 
    if not give them their ancestors default value
    (it fills all the Nones of the tree)"""
    if node.name == 'default' and node.value != 0 and node.value == None:
        updated_value = False
        while not updated_value:
            ancestor_node = node.parent.parent
            selected_child = select_child_with_name(ancestor_node, "default")
            if selected_child.value == None:
                ancestor_node = ancestor_node.parent
            else:
                node.value = selected_child.value
                updated_value = True
    for child in node.children:
        fill_betas(child)
    
    return


def print_node_with_value(node):
    """Custom function to print node names and values if present"""
    if hasattr(node, 'value'):
        print("%s%s" % (node.depth * '  ', node.name), end=':')
        print(" %s" % node.value)
    else:
        print("%s%s" % (node.depth * '  ', node.name))

    return


def create_template_map(spv_nside, nside, use_jax=False):
    """Create one spv template map"""
    print('>>> Creating new template for: ', spv_nside)
    if use_jax:
        # multires case # TODO: implement the case adaptive
        chx.assert_shape(spv_nside, (1,))
        
        def wrapper_ud_grade(nside_in_):
            nside_in = nside_in_[0]
            if nside_in == 0:
                map_out = np.ones(12*nside**2)
            else:
                map_in = np.arange(12*nside_in**2)
                map_out = hp.ud_grade(map_in, nside_out=nside)
            return map_out

        def pure_call_ud_grade(nside_in):
            shape_output = (12*nside**2,)
            return jax.pure_callback(wrapper_ud_grade, jax.ShapeDtypeStruct(shape_output, np.float64), nside_in,)

        spv_template = pure_call_ud_grade(spv_nside)
        
        return spv_template
    
    if len(spv_nside) == 1:
        # multires case
        ns = spv_nside[0]
        if ns == 0:
            spv_template = np.ones(12*nside**2)
        else:
            spv_template = hp.ud_grade(np.arange(12*ns**2), nside_out=nside)
    else:
        # adaptive multires case
        # TODO: implement the case adaptive multires where spv_nside is a list
        NotImplementedError("Only one nside is supported for now")
    
    return spv_template


def build_empty_tree_spv(n_fgs_comp, n_betas):
    """Create empty tree for spv config,
    with 0 in node nside_spv (corresponding to basic comp sep)"""
    root = Node("root")
    nside_spv = Node('nside_spv', parent=root)
    default_nside_spv = Node('default', parent=nside_spv)
    default_nside_spv.value = [0]
    for i in range(n_fgs_comp):
        f_node = Node('f'+str(i+1), parent=nside_spv)
        default_f_node = Node('default', parent=f_node)
        default_f_node.value = None
        for j in range(n_betas//n_fgs_comp):
            b_node = Node('b'+str(j+1), parent=f_node)
            default_b_node = Node('default', parent=b_node)
            default_b_node.value = None

    return root


#### Higher level functions
def tree_spv_config(yaml_file_path, n_betas, n_fgs_comp, print_tree=False):
    """From spv param file to tree of spv config"""
    try:
        open(yaml_file_path, 'r')
    except:
        print('No yaml file found in given path: ', yaml_file_path, flush=True)
        yaml_file_path = ''

    if yaml_file_path != '':
        # Read in dict spv params from .yaml file
        dict_params_spv = read_spv_config(yaml_file_path)

        # From dict to tree
        root = Node("root")
        build_tree_from_dict(dict_params_spv, parent=root)
        
        # Count nodes starting with 'b'
        count_b = count_betas_in_tree(root)
        print('count_b:', count_b)
        print('n_betas: ', n_betas)
        assert count_b == n_betas
    else:
        # create default tree structure 
        # (corresponds to no spatial variability case)
        print('No spatial variability case', flush=True)
        root = build_empty_tree_spv(n_fgs_comp, n_betas)
    
    if print_tree:
        for _, _, node in RenderTree(root):
            print_node_with_value(node)
    
    # Fill nodes w/o value
    fill_betas(root)

    # Print the tree structure with names and values if present
    if print_tree:
        for _, _, node in RenderTree(root):
            print_node_with_value(node)

    return root


def create_one_template_from_bdefaultvalue(nside_b, all_nsides, spv_templates, nside, use_jax=False):
    try:
        idx = all_nsides.index(nside_b)
        spv_template_b = spv_templates[idx]
    except ValueError:
        spv_template_b = create_template_map(nside_b, nside, use_jax=use_jax)
    all_nsides.append(nside_b)

    return spv_template_b


def create_one_template(node, all_nsides, spv_templates, nside):
    nside_b = node.children[0].value
    spv_template_b = create_one_template_from_bdefaultvalue(nside_b, all_nsides, spv_templates, nside)
    
    return spv_template_b


def create_templates_spv_old(node, nside_out, all_nsides=[], spv_templates=[]):
    """Create templates of spatial variability for all betas
    (it creates all the templates at once and keep them in a list)"""
    # loop over betas and create template maps for spv
    if node.name.startswith('b'):
        spv_template_b = create_one_template(node, all_nsides, spv_templates, nside_out)
        spv_templates.append(spv_template_b)
    for child in node.children:
        create_templates_spv_old(child, nside_out, all_nsides, spv_templates)

    return spv_templates


def get_nodes_b(root_tree):
    nodes = []
    for _, _, node in RenderTree(root_tree):
        if node.name.startswith('b'):
            nodes.append(node)
    return nodes
    

def get_n_patches_b(node_b, jax_use=False):
    # TODO: genralize to pathces w kmeans
    if jax_use:
        # node_b expected to be list of default values of nodes n
        n_patches_b = jnp.where(node_b == 0, 1, 12*node_b**2)
        return n_patches_b

    # node_b expected to be list of the nodes b
    patches_config = node_b.children[0].value
    if patches_config == [0]:
        n_patches_b = 1
    elif len(patches_config) == 1:
        # multires but not adaptive case
        n_patches_b = 12*patches_config[0]**2
    else:
        # adaptive multires case
        NotImplementedError("Adaptive multires case not implemented yet")
    return n_patches_b


def get_values_b(nodes_b, n_frequencies, n_components):
    """get default values of b"""
    return np.array([nodes_b[i].children[0].value for i in range((n_frequencies*n_components))])
