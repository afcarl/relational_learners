"""
Classes of Relational Learners that learn in a General to Specific Fashion.
"""
from pprint import pprint

from py_search.base import Problem
from py_search.base import Node
from py_search.optimize import branch_and_bound

from planners.fo_planner import Operator
from planners.fo_planner import build_index
from planners.fo_planner import subst
from planners.fo_planner import extract_strings
from planners.fo_planner import is_variable

from learners.relational.utils import covers
from learners.relational.utils import rename
from learners.relational.utils import generalize_literal
from learners.relational.utils import remove_vars
from learners.relational.utils import clause_length


def specialize(h, constraints, args, pset, neg, neg_mapping, gensym,
               depth_limit=10):
    """
    Returns the set of most general specializations of h that do NOT
    cover x.
    """
    problem = SpecializationProblem(h, extra=(args, constraints, pset, neg,
                                              neg_mapping, gensym))
    sol_set = set()
    for sol in branch_and_bound(problem, depth_limit=depth_limit):
        sol_set.add(sol.state)
        if len(sol_set) >= 25:
            break
    return sol_set


class SpecializationProblem(Problem):

    def goal_test(self, node):
        """
        Treat specialization as an optimization problem. Continue until the
        best node is found.
        """
        return False

    def node_value(self, node):
        """
        The best potential score from this node.
        """
        h = node.state
        p_excluded, n_excluded, pset, nset, gensym = node.extra

        c_length = clause_length(h)

        best_score = len(pset) - (c_length + 1)
        return min(node.cost(), -1 * best_score)

    def successors(self, node):
        h = node.state
        constraints, c_length, pset, nset, gensym = node.extra

        if len(pset) == 0:
            return

        p, pm = choice(pset)
        p_index = build_index(p)

        found = False
        for m in operator.match(p_index, initial_mapping=pm):
            reverse_m = {m[a]: a for a in m}
            pos_partial = set([rename(reverse_m, x) for x in p])
            found = True
            break

        if not found:
            return

        # specialize current variables using pset?
        # add new literals from pset


        new_pset = [(p, pm) for p, pm in pset if
                    covers(new_h.union(constraints), p, pm)]
        new_nset = [(n, nm) for n, nm in pset if
                    covers(new_h.union(constraints), n, nm)]
        new_h_length = c_length + 1

        score = len(new_pset) - (len(new_nset) + new_h_length)

        # yield Node(new_h, node, action, node.cost()+1, node.extra)


class IncrementalHeuristicGS(object):

    def __init__(self, args=None, constraints=None):
        """
        A relational learner that searches in a general to specific fashion.
        It also only maintains a single hypothesis and uses a 

        args - a tuple of arguments to the learner, these are the args in the
        head of the rule.
        constraints - a set of constraints that cannot be removed. These can be
        used to ensure basic things like an FOA must have a value that isn't an
        empty string, etc.
        """
        if args is None:
            args = tuple([])
        if constraints is None:
            constraints = frozenset([])

        self.args = args
        self.constraints = constraints
        self.pset = []
        self.hset = set([frozenset([])])
        self.gen_counter = 0

    def get_hset(self):
        """
        Gets a list of hypotheses. This is essentially a disjunction of
        conjunctions. Each hypothesis can be fed into a pattern matcher to
        perform matching.
        """
        return [h.union(self.constraints) for h in self.hset]

    def ifit(self, t, x, y):
        """
        Incrementally specializes the hypothesis set.
        """
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        if y == 1:
            self.pset.append((x, mapping))
            bad_h = set([h for h in self.hset
                         if not covers(h.union(self.constraints), x, mapping)])
            # print("POS BAD", bad_h)
            self.hset -= bad_h

        elif y == 0:
            bad_h = set([h for h in self.hset
                         if covers(h.union(self.constraints), x, mapping)])
            # print("NEG BAD", bad_h)
            for h in bad_h:
                self.hset.remove(h)
                gset = specialize(h, self.constraints, self.args, self.pset, x,
                                  mapping, lambda: self.gensym())
                for p, pm in self.pset:
                    bad_g = set([g for g in gset if not
                                 covers(g.union(self.constraints), p, pm)])
                    gset -= bad_g
                # print("WORKABLE GSET", gset)
                self.hset.update(gset)

            self.remove_subsumed()

            # impose a limit on the number of hypotheses
            self.hset = set(list(self.hset)[:10])

        else:
            raise Exception("y must be 0 or 1")


class IncrementalGeneralToSpecific(object):

    def __init__(self, args=None, constraints=None):
        """
        A relational learner that searches in a general to specific fashion.
        I try to limit the specialization to keep things tractable...

        args - a tuple of arguments to the learner, these are the args in the
        head of the rule.
        constraints - a set of constraints that cannot be removed. These can be
        used to ensure basic things like an FOA must have a value that isn't an
        empty string, etc.
        """
        if args is None:
            args = tuple([])
        if constraints is None:
            constraints = frozenset([])

        self.args = args
        self.constraints = constraints
        self.pset = []
        self.hset = set([frozenset([])])
        self.gen_counter = 0

    def get_hset(self):
        """
        Gets a list of hypotheses. This is essentially a disjunction of
        conjunctions. Each hypothesis can be fed into a pattern matcher to
        perform matching.
        """
        return [h.union(self.constraints) for h in self.hset]

    def ifit(self, t, x, y):
        """
        Incrementally specializes the hypothesis set.
        """
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        if y == 1:
            self.pset.append((x, mapping))
            bad_h = set([h for h in self.hset
                         if not covers(h.union(self.constraints), x, mapping)])
            # print("POS BAD", bad_h)
            self.hset -= bad_h

        elif y == 0:
            bad_h = set([h for h in self.hset
                         if covers(h.union(self.constraints), x, mapping)])
            # print("NEG BAD", bad_h)
            for h in bad_h:
                self.hset.remove(h)
                gset = specialize(h, self.constraints, self.args, self.pset, x,
                                  mapping, lambda: self.gensym())
                for p, pm in self.pset:
                    bad_g = set([g for g in gset if not
                                 covers(g.union(self.constraints), p, pm)])
                    gset -= bad_g
                # print("WORKABLE GSET", gset)
                self.hset.update(gset)

            self.remove_subsumed()

            # impose a limit on the number of hypotheses
            self.hset = set(list(self.hset)[:10])

        else:
            raise Exception("y must be 0 or 1")

    def remove_subsumed(self):
        """
        Removes hypotheses from the hset that are generalizations of other
        hypotheses in hset.
        """
        bad_h = set()
        hset = list(self.hset)
        for i, h in enumerate(hset):
            if h in bad_h:
                continue
            for g in hset[i+1:]:
                if g in bad_h:
                    continue

                rename_negation = {'not': "--NOT--"}
                rh = frozenset(rename(rename_negation, ele) for ele in h)
                rg = frozenset(rename(rename_negation, ele) for ele in g)

                h_specializes_g = self.is_specialization(rh, rg)
                g_specializes_h = self.is_specialization(rg, rh)

                if h_specializes_g and g_specializes_h:
                    print(h, 'equals', g)
                    if len(h) < len(g):
                        bad_h.add(g)
                    else:
                        bad_h.add(h)

                elif h_specializes_g:
                    bad_h.add(h)
                elif g_specializes_h:
                    bad_h.add(g)

        self.hset -= bad_h

    def is_specialization(self, s, h):
        """
        Takes two hypotheses s and g and returns True if s is a specialization
        of h. Note, it returns False if s and h are equal (s is not a
        specialization in this case).
        """
        if s == h:
            return False

        # remove vars, so the unification isn't going in both directions.
        s = set(remove_vars(l) for l in s)

        # check if h matches s (then s specializes h)
        index = build_index(s)
        operator = Operator(tuple(['Rule']), h, [])
        for m in operator.match(index):
            return True
        return False

    def gensym(self):
        self.gen_counter += 1
        return '?new_gen%i' % self.gen_counter


if __name__ == "__main__":

    p1 = {('color', 'dark'),
          ('tails', '2'),
          ('nuclei', '2'),
          ('wall', 'thin')}
    n1 = {('color', 'light'),
          ('tails', '2'),
          ('nuclei', '1'),
          ('wall', 'thin')}
    p2 = {('color', 'light'),
          ('tails', '2'),
          ('nuclei', '2'),
          ('wall', 'thin')}
    n2 = {('color', 'dark'),
          ('tails', '1'),
          ('nuclei', '2'),
          ('wall', 'thick')}

    X = [p1, n1, p2, n2]
    y = [1, 0, 1, 0]

    learner = IncrementalGeneralToSpecific()

    for i, x in enumerate(X):
        print("Adding the following instance (%i):" % y[i])
        pprint(x)
        learner.ifit(tuple([]), x, y[i])
        print("Resulting hset")
        print(learner.get_hset())
        print(len(learner.get_hset()))
