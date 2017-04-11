"""
Classes of Relational Learners that learn in a General to Specific Fashion.
"""
from pprint import pprint
from random import choice

from py_search.base import Problem
from py_search.base import Node
from py_search.optimization import branch_and_bound
from py_search.optimization import hill_climbing
from py_search.optimization import simulated_annealing

from fo_planner import Operator
from fo_planner import build_index
from fo_planner import subst
from fo_planner import extract_strings
from fo_planner import is_variable

from utils import covers
from utils import rename
from utils import generalize_literal
from utils import remove_vars
from utils import clause_length
from utils import test_coverage
from utils import generate_literal
from utils import count_occurances
from utils import get_variablizations


def clause_score(pos_covered, neg_covered, num_pos, num_neg, length):
    w = 0.95
    accuracy = (pos_covered + (num_neg - neg_covered)) / (num_pos + num_neg)
    return w * accuracy + (1-w) * 1/(1+length)


def generalize(h, constraints, pset, nset, gensym, depth_limit=10):
    """
    Returns the set of most specific generalization of h that do NOT
    cover x.
    """
    c_length = clause_length(h)
    psubset, nsubset = test_coverage(h, constraints, pset, nset)
    p_uncovered = [p for p in pset if p not in psubset]
    n_uncovered = [n for n in nset if n not in nsubset]
    initial_score = clause_score(len(psubset), len(nsubset), len(pset),
                                 len(nset), c_length)
    problem = GeneralizationProblem(h, initial_cost=-1*initial_score,
                                    extra=(constraints, c_length, p_uncovered,
                                           n_uncovered, pset, nset, gensym))
    for sol in branch_and_bound(problem, depth_limit=depth_limit):
        print("SOLUTION FOUND", sol.state)
        return sol.state


class GeneralizationProblem(Problem):

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
        (constraints, c_length, p_uncovered, n_uncovered, pset,
         nset, gensym) = node.extra

        c_length = clause_length(h)
        best_score = self.score(len(pset), len(nset) - len(n_uncovered),
                                len(pset), len(nset), max(0, (c_length - 1)))
        print(h, 'curr=%0.2f' % node.cost(),
              'min=%0.2f' % min(node.cost(), -1 * best_score))
        print(len(pset)-len(p_uncovered), len(nset)-len(n_uncovered),
              len(pset), len(nset))
        return min(node.cost(), -1 * best_score)

    def score(self, pos_covered, neg_covered, num_pos, num_neg, length):
        w = 0.95
        accuracy = ((pos_covered + (num_neg - neg_covered)) /
                    (num_pos + num_neg))
        return w * accuracy + (1-w) * 1/(1+length)

    def successors(self, node):
        h = node.state
        (constraints, c_length, p_uncovered, n_uncovered, pset, nset,
         gensym) = node.extra

        # remove literals
        for literal in h:
            removable = True
            for ele in literal[1:]:
                if not is_variable(ele):
                    removable = False
                    break
                if (count_occurances(ele, h) > 1):
                    removable = False
                    break

            if removable:
                new_h = frozenset(x for x in h if x != literal)
                new_pset, new_nset = test_coverage(new_h, constraints,
                                                   p_uncovered, n_uncovered)
                new_p_uncovered = 
                new_c_length = c_length - 1
                # new_c_length = clause_length(new_h)
                score = self.score(len(new_pset), len(new_nset), len(pset),
                                   len(nset), new_c_length)

                yield Node(new_h, node, ('remove literal', literal), -1 *
                           score, extra=(constraints, new_c_length, new_pset,
                                         new_nset, pset, nset, gensym))

        # replace constants with variables.
        for literal in h:
            for new_l in get_variablizations(literal, gensym):
                new_h = frozenset([x if x != literal else new_l for
                                   x in h])
                new_pset, new_nset = test_coverage(new_h, constraints, pset,
                                                   nset)
                new_c_length = c_length - 1
                # new_c_length = clause_length(new_h)
                score = self.score(len(new_pset), len(new_nset), len(pset),
                                   len(nset), new_c_length)

                yield Node(new_h, node, ('variablize', literal, new_l), -1 *
                           score, extra=(constraints, new_c_length, new_pset,
                                         new_nset, pset, nset, gensym))

        # TODO replace instances of repeating variable with new variable


def specialize(h, constraints, pset, nset, gensym, depth_limit=10):
    """
    Returns the set of most general specializations of h that do NOT
    cover x.
    """
    c_length = clause_length(h)
    problem = SpecializationProblem(h, extra=(constraints, c_length, pset,
                                              nset, gensym))
    for sol in branch_and_bound(problem, depth_limit=depth_limit):
        return sol.state


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
        constraints, c_length, pset, nset, gensym = node.extra

        c_length = clause_length(h)

        # best_score = len(pset) - (c_length + 1)
        best_score = len(pset) - 0.01 * (c_length + 1)
        best_score = self.score(len(pset), 0, (c_length + 1))
        print(h, 'curr=%0.2f' % node.cost(), 'min=%0.2f' % min(node.cost(), -1 * best_score))
        return min(node.cost(), -1 * best_score)

    # def random_successor(self, node):
    #     successors = [n for n in self.successors(node)]
    #     if len(successors) > 0:
    #         return choice(successors)

    def score(self, n_pos, n_neg, length):
        return n_pos - n_neg - 0.01 * length

    def successors(self, node):
        h = node.state
        constraints, c_length, pset, nset, gensym = node.extra

        if len(pset) == 0:
            return

        p, pm = choice(pset)
        p_index = build_index(p)

        operator = Operator(tuple(('Rule',)),
                            h.union(constraints), [])

        found = False
        for m in operator.match(p_index, initial_mapping=pm):
            reverse_m = {m[a]: a for a in m}
            pos_partial = set([rename(reverse_m, x) for x in p])
            found = True
            break

        if not found:
            return

        # specialize current variables using pset?
        for var in m:
            limited_m = {var: m[var]}
            new_h = frozenset([subst(limited_m, l) for l in h])

            new_pset, new_nset = test_coverage(new_h, constraints, pset, nset)
            new_c_length = c_length + 1
            score = self.score(len(new_pset), len(new_nset), 0.01 *
                               new_c_length)

            yield Node(new_h, node, ('specializing var', var, m[var]), -1 * score,
                       extra=(constraints, new_c_length, new_pset,
                              new_nset, gensym))

        # add new literals from pset
        for l in pos_partial:
            if l not in h:
                l = generate_literal(l[0], len(l)-1, gensym)
                # l = generalize_literal(l, gensym)

                new_h = h.union([l])


                new_pset, new_nset = test_coverage(new_h, constraints, pset, nset)

                new_c_length = c_length + 1

                score = self.score(len(new_pset), len(new_nset), 0.01 *
                                   new_c_length)

                yield Node(new_h, node, ('add literal', l), -1 * score,
                           extra=(constraints, new_c_length, new_pset,
                                  new_nset, gensym))


class IncrementalHeuristic(object):

    def __init__(self, args=None, constraints=None):
        """
        A relational learner that searches in a general to specific fashion.
        It also only maintains a single hypothesis.

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
        self.nset = []
        # self.hset = set([frozenset([])])
        self.hset = set()
        self.gen_counter = 0

    def gensym(self):
        self.gen_counter += 1
        return '?new_gen%i' % self.gen_counter

    def get_hset(self):
        """
        Gets a list of hypotheses. This is essentially a disjunction of
        conjunctions. Each hypothesis can be fed into a pattern matcher to
        perform matching.
        """
        return [h.union(self.constraints) for h in self.hset]

    def compute_bottom_clause(self, x, mapping):
        reverse_m = {mapping[a]: a for a in mapping}
        partial = set([rename(reverse_m, l) for l in x])
        return frozenset(partial)

    def score(self, hset, constraints, pset, nset):
        w = 0.5
        np = len(pset)
        nn = len(nset)
        h_length = 1 + sum([clause_length(h) for h in hset])
        p_covered = 0
        n_covered = 0
        for p, pm in pset:
            found = False
            for h in hset:
                if covers(h.union(self.constraints), p, pm):
                    p_covered += 1
                    found = True
                    break
            if found:
                break

        for n, nm in nset:
            found = False
            for h in hset:
                if covers(h.union(self.constraints), n, nm):
                    n_covered += 1
                    found = True
                    break
            if found:
                break

        return (w * ((p_covered + (nn - n_covered)) / (np + nn)) +
                (1-w) * len(hset))


    def ifit(self, t, x, y):
        """
        Incrementally specializes the hypothesis set.
        """
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        if y == 1:
            self.pset.append((x, mapping))

            match = False
            for h in self.hset:
                if covers(h.union(self.constraints), x, mapping):
                    match = True

            # if it fails to cover then try generalizing each hypothesis to
            # improve the fit.
            if not match:
                update_hset = set([generalize(h, self.constraints, self.pset,
                                              self.nset, lambda: self.gensym())
                                   for h in self.hset])
                ms = frozenset([self.compute_bottom_clause(x, mapping)])
                new_hset = self.hset.union(ms)

                update_score = self.score(update_hset, self.constraints,
                                          self.pset, self.nset)
                new_score = self.score(new_hset, self.constraints, self.pset,
                                       self.nset)
                if update_score > new_score:
                    self.hset = update_hset
                else:
                    self.hset = new_hset

            # also try adding a new, specific, hypothesis

            #pick the best

        elif y == 0:
            self.nset.append((x, mapping))
            bad_h = set([h for h in self.hset
                         if covers(h.union(self.constraints), x, mapping)])
            # print("NEG BAD", bad_h)

            update_hset = set([specialize(h, self.constraints, self.pset,
                                          self.nset, lambda: self.gensym()) if
                               h in bad_h else h for h in self.hset])
            remove_hset = set([h for h in self.hset if h not in bad_h])

            update_score = self.score(update_hset, self.constraints,
                                      self.pset, self.nset)
            remove_score = self.score(remove_hset, self.constraints, self.pset,
                                      self.nset)
            if update_score > remove_score:
                self.hset = update_hset
            else:
                self.hset = remove_hset

        else:
            raise Exception("y must be 0 or 1")

        print("OVERALL SCORE", self.score(self.hset, self.constraints,
                                          self.pset, self.nset))


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

    learner = IncrementalHeuristic()

    for i, x in enumerate(X):
        print("Adding the following instance (%i):" % y[i])
        pprint(x)
        learner.ifit(tuple([]), x, y[i])
        print("Resulting hset")
        print(learner.get_hset())
        print(len(learner.get_hset()))
