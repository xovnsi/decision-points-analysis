from utils import get_in_place_loop, get_out_place_loop
from utils import get_loop_not_silent, get_input_near_source


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
        self.active = True

    def set_inactive(self):
        self.active = False

    def is_active(self):
        return self.active
    
    def is_vertex_in_loop(self, vertex):
        is_in_loop = False
        if vertex in self.vertex:
            is_in_loop = True
        return is_in_loop

    def is_vertex_input_loop(self, vertex):
        is_input_loop = False
        if vertex in self.input_places:
            is_input_loop = True
        return is_input_loop

    def is_vertex_nearest_input(self, vertex):
        is_nearest = False
        if vertex == self.nearest:
            is_nearest = True
        return is_nearest

    def is_vertex_output_loop(self, vertex):
        is_output_loop = False
        if vertex in self.output_places:
            is_output_loop = True
        return is_output_loop

    def set_nearest_input(self, net, loops):
        self.nearest = get_input_near_source(net, self.input_places.copy(), loops)

    def set_dp_forward_order_transition(self, net):
        dp_forward = dict()
        places = [place.name for place in net.places]
        for vertex in self.vertex:
            if vertex in places:
                #breakpoint()
                place = [place for place in net.places if place.name == vertex][0]
                if len(place.out_arcs) > 1:
                    not_silent = set()
                    dp_forward[place.name] = dict()
                    for out_arc in place.out_arcs:
                        if out_arc.target.label is None:
                            for out_arc_inn in out_arc.target.out_arcs:
                                #breakpoint()
                                not_silent = self.get_next_not_silent_forward(out_arc_inn.target, not_silent)
                        dp_forward[place.name][out_arc.target.name] = not_silent
        self.dp_forward = dp_forward


    def get_next_not_silent_forward(self, place, not_silent) -> list:
        """ Recursively compute the first not silent transition connected to a place

        Given a place and a list of not silent transition (i.e. without label) computes
        the next not silent transitions in order to correctly characterize the path through
        the considered place. The algorithm stops in presence of a joint-node (if not in a loop) 
        or when all of the output transitions are not silent. If at least one transition is 
        silent, the algorithm computes recursively the next not silent.
        """
        # first stop condition
        #breakpoint()
        if place.name == 'sink':
            return not_silent
        is_input_a_skip = len(place.in_arcs) == 2 and len([arc.source.name for arc in place.in_arcs if arc.source.label is None]) == 1
        if (len(place.in_arcs) > 1 and not (is_input_a_skip or self.is_vertex_input_loop(place.name))) or self.is_vertex_nearest_input(place.name): # or (is_output and loop_name == loop_name_start):
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
        dp_backward = dict()
        places = [place.name for place in net.places]
        for vertex in self.vertex:
            if vertex in places:
                place = [place for place in net.places if place.name == vertex][0]
                if self.check_if_place_reachable_through_inv_act(place, self.nearest, False):
                    nearest = [place for place in net.places if place.name == self.nearest][0]
                    if len(place.out_arcs) > 1:
                        not_silent = set()
                        dp_backward[place.name] = dict()
                        for out_arc in place.out_arcs:
                            if out_arc.target.label is None:
                                for out_arc_inn in out_arc.target.out_arcs:
                                    #breakpoint()
                                    not_silent = self.get_next_not_silent_backward(nearest, not_silent, place.name)
                                dp_backward[place.name][out_arc.target.name] = not_silent
            self.dp_backward = dp_backward

    def get_next_not_silent_backward(self, place, not_silent, end_place) -> list:
        """ Recursively compute the first not silent transition connected to a place

        Given a place and a list of not silent transition (i.e. without label) computes
        the next not silent transitions in order to correctly characterize the path through
        the considered place. The algorithm stops in presence of a joint-node (if not in a loop) 
        or when all of the output transitions are not silent. If at least one transition is 
        silent, the algorithm computes recursively the next not silent.
        """
        # first stop condition
        #breakpoint()
        if place.name == 'sink':
            return not_silent
        is_input_a_skip = len(place.in_arcs) == 2 and len([arc.source.name for arc in place.in_arcs if arc.source.label is None]) == 1
        if (len(place.in_arcs) > 1 and not (is_input_a_skip or self.is_vertex_input_loop(place.name))) or place.name == end_place: # or (is_output and loop_name == loop_name_start):
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
                        not_silent = self.get_next_not_silent_backward(out_arc_inn.target, not_silent, end_place)
        return not_silent

    def check_if_loop_is_active(self, net, sequence):
        #breakpoint()
        if not sequence[-1] in self.vertex:
            self.set_inactive()
            self.number_of_loops = 0
        elif not self.is_active():
            if len(sequence) > 1:
                if sequence[-2] in self.vertex:
                    trans_obj = [trans_net for trans_net in net.transitions if trans_net.name == sequence[-2]][0]
                    #breakpoint()
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
        # TODO it is possible to check only the last two of the sequence?
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
                if not reachable:
                    self.number_of_loops += 1

    def check_if_reachable(self, start_trans, end_trans, reachable):
        for out_arc in start_trans.out_arcs:
            if out_arc.target.name == "sink" or out_arc.target.name == self.nearest:
                return reachable
            else:
                for out_arc_inn in out_arc.target.out_arcs:
                    if out_arc_inn.target.name in self.vertex:
                        if out_arc_inn.target.label is None:
                            reachable = self.check_if_reachable(out_arc_inn.target, end_trans, reachable)
                        else:
                            if out_arc_inn.target.name == end_trans:
                                reachable = True
                    if reachable:
                        break
            if reachable:
                break
        return reachable

    def check_if_place_reachable_through_inv_act(self, start_place, end_place, reachable):
        for out_arc in start_place.out_arcs:
            if not out_arc.target.label is None:
                return reachable
            else:
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
