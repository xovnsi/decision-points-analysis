import numpy as np
import operator
import pandas as pd
from typing import TypeVar

TypeNode = TypeVar('TypeNode', bound='Node')


class Node(object):
    """ Class implementing a generic node of a decision tree """

    def __init__(self, parent_attribute_value, parent_level):
        self._label = parent_attribute_value        # name of the node. Corresponds to the condition to be fulfilled in the parent node
        self._childs = set()
        self._attribute = None                      # attribute the node splits on. For continuous attributes the format is: "attribute:threshold"
        self._attribute_type = None                 # type of the attribute ['categorical', 'boolean', 'continuous']
        self._parent_node = None
        self._level = parent_level + 1

    def set_attribute(self, attribute, attribute_type) -> None:
        """ Sets the attribute (and its type) on which the node splits """
        self._attribute = attribute
        self._attribute_type = attribute_type
        if self._attribute_type == 'continuous':
            try:
                self._threshold = float(self._attribute.split(':')[1])
            except:
                raise Exception("Threshold must be a numerical value")
        elif not self._attribute_type in ['categorical', 'boolean']:
            raise Exception("Attribute value not supported")

    def get_level(self) -> int:
        return self._level

    def get_label(self) -> str:
        """ Returns the name of the node """
        return self._label

    def get_attribute(self) -> str:
        """ Returns the attribute of the node """
        return self._attribute

    def get_parent_node(self) -> TypeNode:
        """ Get the node's parent """
        return self._parent_node

    def set_parent_node(self, parent_node) -> None:
        """ Set the node's parent """
        self._parent_node = parent_node

class DecisionNode(Node):
    """ Class implementing a decision node of a decision tree """

    def get_childs(self) -> set:
        """ Returns the set of the node childs """
        return self._childs

    def add_child(self, child) -> None:
        """ Adds another node to the set of node childs """
        if child == self:
            raise Exception("A node can't have itself as child.")
        self._childs.add(child)

    def delete_child(self, child) -> None:
        """ Removes a node from the set of node childs """
        self._childs.remove(child)

    def run_test(self, attr_value) -> str:
        """ Returns the condition given by the attribute value """
        attr_name = self._attribute.split(':')[0]
        if self._attribute_type == 'continuous':
            if self.continuous_test_function_lower(attr_value):
                result_test = "{} <= {}".format(attr_name, self._threshold)
            else:
                result_test = "{} > {}".format(attr_name, self._threshold)
        else:
            result_test = "{} = {}".format(attr_name, attr_value)
        return result_test

    def get_child(self, attr_value) -> TypeNode:
        """ Returns the child fulfilling the condition given by the attribute value """
        return next((child for child in self._childs if child.get_label() == self.run_test(attr_value)), None)

    def continuous_test_function_lower(self, attr_value) -> bool:
        """ Test if the attribute value is less than the threshold """
        return attr_value <= self._threshold


class LeafNode(Node):
    def __init__(self, classes, parent_attribute_value, parent_level):
        super().__init__(parent_attribute_value, parent_level)
        self._classes = classes
        # The label class of the node is the class with maximum number of examples
        self._label_class = max(self._classes.items(), key=operator.itemgetter(1))[0] 

    def get_class_names(self) -> list:
        """ Returns the classes contained in the node after the training """
        return list(self._classes.keys())

    def get_class_examples(self, class_name) -> int:
        """ Retruns the number of examples of a specific class contained in the node after the training """
        return self._classes[class_name]
    
