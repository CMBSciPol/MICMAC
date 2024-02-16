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
    if node.name == 'default' and not node.value:
        updated_value = False
        while not updated_value:
            ancestor_node = node.parent.parent
            selected_child = select_child_with_name(ancestor_node, "default")
            if not selected_child.value:
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


def create_template_map(spv_nside, nside_out):
    """Create one spv template map"""
    print('>>> Creating new template for: ', spv_nside)
    if len(spv_nside) == 1:
        # multires case
        ns = spv_nside[0]
        spv_template = hp.ud_grade(np.arange(12*ns**2), nside_out=nside_out)
    else:
        # adaptive multires case
        # TODO: implement the case adaptive multires where spv_nside is a list
        NotImplementedError("Only one nside is supported for now")
    return spv_template



#### Higher level functions
def tree_spv_config(yaml_file_path, n_betas, print_tree=False):
    """From spv param file to tree of spv config"""
    # Read in dict spv params from .yaml file
    dict_params_spv = read_spv_config(yaml_file_path)

    # From dict to tree
    root = Node("root")
    build_tree_from_dict(dict_params_spv, parent=root)

    # Count nodes starting with 'b'
    count_b = count_betas_in_tree(root)
    print(count_b)
    assert count_b == n_betas

    # Fill nodes w/o value
    fill_betas(root)

    # Print the tree structure with names and values if present
    if print_tree:
        for _, _, node in RenderTree(root):
            print_node_with_value(node)

    return root


def create_one_template(node, all_nsides, spv_templates, nside_out):
    nside_b = node.children[0].value
    try:
        idx = all_nsides.index(nside_b)
        spv_template_b = spv_templates[idx]
    except ValueError:
        spv_template_b = create_template_map(nside_b, nside_out)
    all_nsides.append(nside_b)
    
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
    

def get_n_patches_b(node_b):
    # TODO: genralize to pathces w kmeans
    patches_config = node_b.children[0].value
    if len(patches_config) == 1:
        # multires but not adaptive case
        n_patches_b = 12*patches_config[0]**2
    else:
        # adaptive multires case
        NotImplementedError("Adaptive multires case not implemented yet")
    return n_patches_b
