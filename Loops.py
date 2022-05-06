from loop_utils import get_in_place_loop, get_out_place_loop
from loop_utils import get_loop_not_silent, get_input_near_source


class Loop(object):
    def __init__(self, vertex_in_loop, events_in_loop, net, name):
        self.name = name
        self.vertex = vertex_in_loop
        self.events = events_in_loop
        self.input_places = get_in_place_loop(net, self.vertex)
        self.output_places = get_out_place_loop(net, self.vertex)
        self.active = False
        self.not_silent_trans = get_loop_not_silent(net, self.vertex)
        self.number_of_loops = 0

    def set_active(self):
        """ Sets the loop active """
        self.active = True

    def set_inactive(self):
        """ Sets the loop inactive """
        self.active = False

    def is_active(self) -> bool:
        """ Returns if the loop is active """
        return self.active
    
    def is_vertex_in_loop(self, vertex) -> bool:
        """ Checks if a node is in the loop """
        is_in_loop = False
        if vertex in self.vertex:
            is_in_loop = True
        return is_in_loop

    def is_vertex_input_loop(self, vertex) -> bool:
        """ Returns if a node is an input node of the loop """
        is_input_loop = False
        if vertex in self.input_places:
            is_input_loop = True
        return is_input_loop

    def is_vertex_nearest_input(self, vertex) -> bool:
        """ Returns if a node is the input node nearest to the net source """
        is_nearest = False
        if vertex == self.nearest:
            is_nearest = True
        return is_nearest

    def is_vertex_output_loop(self, vertex) -> bool:
        """ Returns if a node is an output node of the loop """
        is_output_loop = False
        if vertex in self.output_places:
            is_output_loop = True
        return is_output_loop

    def set_nearest_input(self, net, loops):
        """ Sets the nearest input to the net source """
        self.nearest = get_input_near_source(net, self.input_places.copy(), loops)

    def set_dp_forward_order_transition(self, net):
        """ Sets the forward path not silent transitions

        For every decision point sets the not silent transitions reachable from the decision point
        to the nearest input node of the loop
        """
        dp_forward = dict()
        places = [place.name for place in net.places]
        for vertex in self.vertex:
            # only for places of the net
            if vertex in places:
                place = [place for place in net.places if place.name == vertex][0]
                # only if it is a decision point
                if len(place.out_arcs) > 1:
                    not_silent = set()
                    dp_forward[place.name] = dict()
                    for out_arc in place.out_arcs:
                        # if it is an invisible transition
                        if out_arc.target.label is None:
                            for out_arc_inn in out_arc.target.out_arcs:
                                not_silent = self.get_next_not_silent_forward(out_arc_inn.target, not_silent)
                            dp_forward[place.name][out_arc.target.name] = not_silent
        self.dp_forward = dp_forward

    def get_next_not_silent_forward(self, place, not_silent) -> list:
        """ Recursively compute the first not silent transitions connected to a decision point in the forward path

        Given a place and a list of not silent transition (i.e. with label) computes
        the next not silent transitions in order to correctly characterize the path through
        the considered place. The algorithm stops in presence of a joint-node (if not in a "skip", 
        i.e. two activities one silent and one not coming from the same place and going to the same place),
        when all of the output transitions are not silent, when the nearest input of the loop is reached or 
        when the sink of the net is reached. If at least one transition is silent, the algorithm computes
        recursively the next not silent.
        """
        # first stop condition
        if place.name == 'sink':
            return not_silent
        # TODO verify also that they are coming from the same node
        is_input_a_skip = len(place.in_arcs) == 2 and len([arc.source.name for arc in place.in_arcs if arc.source.label is None]) == 1
        if (len(place.in_arcs) > 1 and not (is_input_a_skip or self.is_vertex_input_loop(place.name))) or self.is_vertex_nearest_input(place.name):
            return not_silent
        out_arcs_label = [arc.target.label for arc in place.out_arcs]
        # second stop condition
        if not None in out_arcs_label:
            not_silent = not_silent.union(set(out_arcs_label))
            return not_silent
        for out_arc in place.out_arcs:
            # add not silent
            if out_arc.target.name in self.vertex:
                if not out_arc.target.label is None:
                    not_silent.add(out_arc.target.label)
                else:
                    # recursive part if is silent
                    for out_arc_inn in out_arc.target.out_arcs:
                        not_silent = self.get_next_not_silent_forward(out_arc_inn.target, not_silent)
        return not_silent
    
    def set_dp_backward_order_transition(self, net):
        """ Sets the backward path not silent transitions

        For every decision point sets the not silent transitions reachable from the nearest input of
        the loop to the decision point considered
        """
        dp_backward = dict()
        places = [place.name for place in net.places]
        for vertex in self.vertex:
            # only for places of the net
            if vertex in places:
                place = [place for place in net.places if place.name == vertex][0]
                # do the calculations only if the nearest input is reachable from the considered place
                # through invisible activities
                if self.check_if_place_reachable_through_inv_act(place, self.nearest, False):
                    nearest = [place for place in net.places if place.name == self.nearest][0]
                    # if is a decision point
                    if len(place.out_arcs) > 1:
                        not_silent = set()
                        dp_backward[place.name] = dict()
                        for out_arc in place.out_arcs:
                            # if invisible transition, get next not silent
                            if out_arc.target.label is None:
                                for out_arc_inn in out_arc.target.out_arcs:
                                    not_silent = self.get_next_not_silent_backward(nearest, not_silent, place.name)
                                dp_backward[place.name][out_arc.target.name] = not_silent
            self.dp_backward = dp_backward

    def get_next_not_silent_backward(self, place, not_silent, end_place) -> list:
        """ Recursively compute the first not silent transitions connected to a decision point in the backward path

        Given a place and a list of not silent transition (i.e. without label) computes
        the next not silent transitions in order to correctly characterize the path through
        the considered place. The algorithm stops in presence of a joint-node (if not in a "skip", 
        i.e. two activities one silent and one not coming from the same place and going to the same place),
        when all of the output transitions are not silent, when the end_place is reached or 
        when the sink of the net is reached. If at least one transition is silent, the algorithm computes
        recursively the next not silent.
        """
        # first stop condition
        if place.name == 'sink':
            return not_silent
        is_input_a_skip = len(place.in_arcs) == 2 and len([arc.source.name for arc in place.in_arcs if arc.source.label is None]) == 1
        # second stop condition
        if (len(place.in_arcs) > 1 and not (is_input_a_skip or self.is_vertex_input_loop(place.name))) or place.name == end_place: 
            return not_silent
        out_arcs_label = [arc.target.label for arc in place.out_arcs]
        # third stop condition
        if not None in out_arcs_label:
            not_silent = not_silent.union(set(out_arcs_label))
            return not_silent
        for out_arc in place.out_arcs:
            # add not silent
            if out_arc.target.name in self.vertex:
                if not out_arc.target.label is None:
                    not_silent.add(out_arc.target.label)
                else:
                    # recursive part if is silent
                    for out_arc_inn in out_arc.target.out_arcs:
                        not_silent = self.get_next_not_silent_backward(out_arc_inn.target, not_silent, end_place)
        return not_silent

    def check_if_loop_is_active(self, net, sequence):
        """ Checks if a loop is active

        A loop is considered active if one activity is not reachable from the previous one
        without passing another time to the input of the loop
        """
        # if the last activity is not in the loop is not active
        if not sequence[-1] in self.vertex:
            self.set_inactive()
            self.number_of_loops = 0
        elif not self.is_active():
            # sequence must be longer than one
            if len(sequence) > 1:
                # also the second penultimate activity must be in the loop
                if sequence[-2] in self.vertex:
                    trans_obj = [trans_net for trans_net in net.transitions if trans_net.name == sequence[-2]][0]
                    # check if the last activity is reachable from the penultimate one
                    reachable = self.check_if_reachable(trans_obj, sequence[-1], False)
                    if not reachable:
                        self.set_active()
                    else:
                        self.set_inactive()
                        self.number_of_loops = 0
                else:
                    self.set_inactive()
                    self.number_of_loops = 0
                    
    def count_number_of_loops(self, net, sequence):
        """ Counts the number of cycle did in the loop based on the sequence

        Given a sequence and the net, it checks how many activities (backward starting from the last)
        are inside the loop and inside that sequence how many activities are not reachable from the previous one
        without passing another time through the input node
        """
        # TODO it is possible to check only the last two of the sequence? LINKED with check_if_active
        sequence.reverse()
        for i in range(len(sequence)):
            if not sequence[i] in self.vertex:
                seq_in_loop_idx = i
                break
        seq_in_loop = sequence[:i]
        seq_in_loop.reverse()
        for i, trans in enumerate(seq_in_loop):
            if i+1 != len(seq_in_loop):
                trans_obj = [trans_net for trans_net in net.transitions if trans_net.name == trans][0]
                reachable = self.check_if_reachable(trans_obj, seq_in_loop[i+1], False)
                # TODO adding directly to the number of loops can be a problem? 
                # if i have already counted the loop at t, going through the sequence at t+1 can wrongly add other cycles?
                # LINKED with check_if_active
                if not reachable:
                    self.number_of_loops += 1

    def check_if_reachable(self, start_trans, end_trans, reachable):
        """ Check if a transition is reachable from another one without passing from the input node """
        for out_arc in start_trans.out_arcs:
            # stops if reaches the net sink or the nearest input node to the net source
            if out_arc.target.name == "sink" or out_arc.target.name == self.nearest:
                return reachable
            else:
                for out_arc_inn in out_arc.target.out_arcs:
                    if out_arc_inn.target.name in self.vertex:
                        #  recursive part if is silent 
                        if out_arc_inn.target.label is None:
                            reachable = self.check_if_reachable(out_arc_inn.target, end_trans, reachable)
                        # the end_trans must be not silent otherwise wouldn't be in the trace
                        else:
                            if out_arc_inn.target.name == end_trans:
                                reachable = True
                    if reachable:
                        break
            if reachable:
                break
        return reachable

    def check_if_place_reachable_through_inv_act(self, start_place, end_place, reachable):
        """ Checks if a place is reachable from another place using invisible transitions

        Recursively check if a place is reachable from another one, through invisible 
        transitions and remaining in the loop.
        """
        for out_arc in start_place.out_arcs:
            if not out_arc.target.label is None:
                return reachable
            # recursive part if is silent 
            else:
                # must be in the loop 
                if out_arc.target.name in self.vertex:
                    for out_arc_inn in out_arc.target.out_arcs:
                        if out_arc_inn.target.name in self.vertex:
                            if out_arc_inn.target.name == end_place:
                                reachable = True
                            else:
                                reachable = self.check_if_place_reachable_through_inv_act(out_arc_inn.target, end_place, reachable)
                        if reachable:
                            break
            if reachable:
                break
        return reachable
