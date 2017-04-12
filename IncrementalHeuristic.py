"""
Classes of Relational Learners that learn in a General to Specific Fashion.
"""
from pprint import pprint
from random import choice

from py_search.base import Problem
from py_search.base import Node
from py_search.optimization import branch_and_bound
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
from utils import count_elements

clause_accuracy_weight = 0.95


def clause_score(accuracy_weight, p_covered, p_uncovered, n_covered,
                 n_uncovered, length):
    w = accuracy_weight
    accuracy = ((p_covered + n_uncovered) / (p_covered + p_uncovered +
                                             n_covered + n_uncovered))
    return w * accuracy + (1-w) * 1/(1+length)


def build_clause(v, possible_literals):
    return frozenset([possible_literals[i][j] for i, j in enumerate(v)])


def optimize_clause(h, constraints, pset, nset, gensym):
    """
    Returns the set of most specific generalization of h that do NOT
    cover x.
    """
    c_length = clause_length(h)
    p_covered, n_covered = test_coverage(h, constraints, pset, nset)
    p_uncovered = [p for p in pset if p not in p_covered]
    n_uncovered = [n for n in nset if n not in n_covered]
    initial_score = clause_score(clause_accuracy_weight, len(p_covered),
                                 len(p_uncovered), len(n_covered),
                                 len(n_uncovered), c_length)
    if len(p_covered) > 0:
        p, pm = choice(p_covered)
        p_index = build_index(p)

        operator = Operator(tuple(('Rule',)),
                            h.union(constraints), [])

        for m in operator.match(p_index, initial_mapping=pm):
            reverse_m = {m[a]: a for a in m}
            pos_partial = set([rename(reverse_m, x) for x in p])
            print('POS PARTIAL', pos_partial)
            break

    # TODO if we wanted we could add the introduction of new variables to the
    # get_variablizations function.
    possible_literals = {}
    for i, l in enumerate(pos_partial):
        possible_literals[i] = [None, l] + [v for v in get_variablizations(l)]
    reverse_pl = {l: (i, j) for i in possible_literals for j, l in
                  enumerate(possible_literals[i])}

    curr = [0 for i in range(len(possible_literals))]
    for l in h:
        i, j = reverse_pl[l]
        curr[i] = j

    print('INITIAL SCORE', initial_score)
    problem = ClauseOptimizationProblem(curr, initial_cost=-1*initial_score,
                                        extra=(possible_literals, constraints,
                                               pset, nset))
    for sol in simulated_annealing(problem, temp_length=30):
        print("SOLUTION FOUND", sol.state)
        return sol.state


class ClauseOptimizationProblem(Problem):

    def goal_test(self, node):
        """
        This is an optimization, so no early termination
        """
        return False

    def simple_random_successor(self, node):
        s = [c for c in self.successors(node)]
        if len(s) > 0:
            return choice(s)

    def random_successor(self, node):
        h = node.state
        (constraints, c_length, p_covered, p_uncovered, n_covered, n_uncovered,
         gensym) = node.extra

        new_hs = []

        ####################################
        # Generate possible specializations
        ####################################
        pos_partial = None

        if len(p_covered) > 0:
            p, pm = choice(p_covered)
            p_index = build_index(p)

            operator = Operator(tuple(('Rule',)),
                                h.union(constraints), [])

            for m in operator.match(p_index, initial_mapping=pm):
                reverse_m = {m[a]: a for a in m}
                pos_partial = set([rename(reverse_m, x) for x in p])
                # print('POS PARTIAL', pos_partial)
                break

        if pos_partial is not None:

            # add new literal
            for l in pos_partial:
                if l not in h:
                    l = generate_literal(l[0], len(l)-1, gensym)
                    # l = generalize_literal(l, gensym)

                    new_h = h.union([l])
                    new_hs.append(('spec', new_h))

            # replace unique var with existing var
            possibly_equal = set()
            for var1 in m:
                for var2 in m:
                    if var1 == var2:
                        continue
                    if m[var1] == m[var2]:
                        pe = frozenset([var1, var2])
                        possibly_equal.add(pe)

            for pe in possibly_equal:
                pe = list(pe)
                rm = {pe[0]: pe[1]}
                new_h = frozenset([rename(rm, l) for l in h])
                new_hs.append(('spec', new_h))

            # TODO EACH OCCURANCE, not each var
            # replace each occurance of existing var with constant
            for var in m:
                limited_m = {var: m[var]}
                new_h = frozenset([subst(limited_m, l) for l in h])
                new_hs.append(('spec', new_h))


        ####################################
        # Generate possible generalizations
        ####################################
        # Remove literal
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
                new_hs.append(('gen', new_h))

        # replace constants with variables.
        for literal in h:
            for new_l in get_variablizations(literal, gensym):
                new_h = frozenset([x if x != literal else new_l for
                                   x in h])
                new_hs.append(('gen', new_h))

        ############################################
        # Choose a new H and generate a node for it
        ############################################
        op, new_h = choice(new_hs)

        if op == "spec":
            new_p_subset, new_n_subset = test_coverage(new_h, constraints,
                                                       p_covered, n_covered)
            new_p_covered = new_p_subset
            new_p_uncovered = p_uncovered + [p for p in p_covered if p not in
                                             new_p_subset]
            new_n_covered = new_n_subset
            new_n_uncovered = n_uncovered + [n for n in n_covered if n not in
                                             new_n_subset]
            new_c_length = c_length + 1
            score = self.score(len(new_p_covered), len(new_p_uncovered),
                               len(new_n_covered), len(new_n_uncovered),
                               new_c_length)

            return Node(new_h, None, None, -1 * score,
                       extra=(constraints, new_c_length, new_p_covered,
                              new_p_uncovered, new_n_covered,
                              new_n_uncovered, gensym))

        elif op == 'gen':
            new_pc_subset, new_nc_subset = test_coverage(new_h,
                                                         constraints,
                                                         p_uncovered,
                                                         n_uncovered)
            new_p_covered = p_covered + new_pc_subset
            new_n_covered = n_covered + new_nc_subset
            new_p_uncovered = [p for p in p_uncovered if p not in
                               new_pc_subset]
            new_n_uncovered = [n for n in n_uncovered if n not in
                               new_nc_subset]
            new_c_length = c_length - 1
            score = self.score(len(new_p_covered), len(new_p_uncovered),
                               len(new_n_covered), len(new_n_uncovered),
                               new_c_length)

            return Node(new_h, None, None, -1 * score,
                       extra=(constraints, new_c_length, new_p_covered,
                              new_p_uncovered, new_n_covered,
                              new_n_uncovered, gensym))

    def score(self, p_covered, p_uncovered, n_covered, n_uncovered, length):
        return clause_score(clause_accuracy_weight, p_covered,
                            p_uncovered, n_covered, n_uncovered, length)

    def gen_specializations(self, node):
        h = node.state
        (constraints, c_length, p_covered, p_uncovered, n_covered, n_uncovered,
         gensym) = node.extra

        if len(p_covered) == 0:
            return

        p, pm = choice(p_covered)
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

            new_p_subset, new_n_subset = test_coverage(new_h, constraints,
                                                       p_covered, n_covered)
            new_p_covered = new_p_subset
            new_p_uncovered = p_uncovered + [p for p in p_covered if p not in
                                             new_p_subset]
            new_n_covered = new_n_subset
            new_n_uncovered = n_uncovered + [n for n in n_covered if n not in
                                             new_n_subset]
            new_c_length = c_length + 1
            score = self.score(len(new_p_covered), len(new_p_uncovered),
                               len(new_n_covered), len(new_n_uncovered),
                               new_c_length)

            yield Node(new_h, None, None, -1 * score,
                       extra=(constraints, new_c_length, new_p_covered,
                              new_p_uncovered, new_n_covered,
                              new_n_uncovered, gensym))

        # add new literals from pset
        for l in pos_partial:
            if l not in h:
                l = generate_literal(l[0], len(l)-1, gensym)
                # l = generalize_literal(l, gensym)

                new_h = h.union([l])

                new_p_subset, new_n_subset = test_coverage(new_h, constraints,
                                                           p_covered,
                                                           n_covered)
                new_p_covered = new_p_subset
                new_p_uncovered = p_uncovered + [p for p in p_covered if p not
                                                 in new_p_subset]
                new_n_covered = new_n_subset
                new_n_uncovered = n_uncovered + [n for n in n_covered if n not
                                                 in new_n_subset]
                new_c_length = c_length + 1
                score = self.score(len(new_p_covered), len(new_p_uncovered),
                                   len(new_n_covered), len(new_n_uncovered),
                                   new_c_length)

                yield Node(new_h, None, None, -1 * score,
                           extra=(constraints, new_c_length, new_p_covered,
                                  new_p_uncovered, new_n_covered,
                                  new_n_uncovered, gensym))

    def gen_generalizations(self, node):
        h = node.state
        (constraints, c_length, p_covered, p_uncovered, n_covered, n_uncovered,
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
                new_pc_subset, new_nc_subset = test_coverage(new_h,
                                                             constraints,
                                                             p_uncovered,
                                                             n_uncovered)
                new_p_covered = p_covered + new_pc_subset
                new_n_covered = n_covered + new_nc_subset
                new_p_uncovered = [p for p in p_uncovered if p not in
                                   new_pc_subset]
                new_n_uncovered = [n for n in n_uncovered if n not in
                                   new_nc_subset]
                new_c_length = c_length - 1
                score = self.score(len(new_p_covered), len(new_p_uncovered),
                                   len(new_n_covered), len(new_n_uncovered),
                                   new_c_length)

                yield Node(new_h, None, None, -1 * score,
                           extra=(constraints, new_c_length, new_p_covered,
                                  new_p_uncovered, new_n_covered,
                                  new_n_uncovered, gensym))

        # replace constants with variables.
        for literal in h:
            for new_l in get_variablizations(literal, gensym):
                new_h = frozenset([x if x != literal else new_l for
                                   x in h])
                new_pc_subset, new_nc_subset = test_coverage(new_h,
                                                             constraints,
                                                             p_uncovered,
                                                             n_uncovered)
                new_p_covered = p_covered + new_pc_subset
                new_n_covered = n_covered + new_nc_subset
                new_p_uncovered = [p for p in p_uncovered if p not in
                                   new_pc_subset]
                new_n_uncovered = [n for n in n_uncovered if n not in
                                   new_nc_subset]
                new_c_length = c_length - 1
                score = self.score(len(new_p_covered), len(new_p_uncovered),
                                   len(new_n_covered), len(new_n_uncovered),
                                   new_c_length)

                yield Node(new_h, None, None, -1 * score,
                           extra=(constraints, new_c_length, new_p_covered,
                                  new_p_uncovered, new_n_covered,
                                  new_n_uncovered, gensym))

    def successors(self, node):

        for child in self.gen_specializations(node):
            yield child

        for child in self.gen_generalizations(node):
            yield child


class IncrementalHeuristic(object):

    def __init__(self, args=None, constraints=None):
        """
        A relational learner that searches the space of hypotheses locally.
        Whenever it receives a new positive or negative example it tries to
        further optimize its hypothesis.

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
        self.h = None
        # self.h = frozenset([])
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
        if self.h is None:
            return []
        return [self.h.union(self.constraints)]

    def compute_bottom_clause(self, x, mapping):
        reverse_m = {mapping[a]: a for a in mapping}
        print("REVERSEM", reverse_m)
        partial = set([rename(reverse_m, l) for l in x])
        return frozenset(partial)

    def ifit(self, t, x, y):
        """
        Incrementally specializes the hypothesis set.
        """
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        if y == 1:
            self.pset.append((x, mapping))
        elif y == 0:
            self.nset.append((x, mapping))
        else:
            raise Exception("y must be 0 or 1")

        if self.h is None and y == 1:
            self.h = self.compute_bottom_clause(x, mapping)
            print("ADDING BOTTOM", self.h)

        if self.h is not None:
            self.h = optimize_clause(self.h, self.constraints, self.pset,
                                     self.nset, lambda: self.gensym())
            c_length = clause_length(self.h)
            p_covered, n_covered = test_coverage(self.h, self.constraints,
                                                 self.pset, self.nset)
            p_uncovered = [p for p in self.pset if p not in p_covered]
            n_uncovered = [n for n in self.nset if n not in n_covered]
            score = clause_score(clause_accuracy_weight, len(p_covered),
                                 len(p_uncovered), len(n_covered),
                                 len(n_uncovered), c_length)

            print("OVERALL SCORE", score)


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

    # for i, x in enumerate(X):
    #     print("Adding the following instance (%i):" % y[i])
    #     pprint(x)
    #     learner.ifit(tuple([]), x, y[i])
    #     print("Resulting hset")
    #     print(learner.get_hset())
    #     print(len(learner.get_hset()))

    p1 = {('person', 'a'),
          ('person', 'b'),
          ('person', 'c'),
          ('parent', 'a', 'b'),
          ('parent', 'b', 'c')}

    n1 = {('person', 'a'),
          ('person', 'b'),
          ('person', 'f'),
          ('person', 'g'),
          ('parent', 'a', 'b'),
          ('parent', 'f', 'g')}

    p2 = {('person', 'f'),
          ('person', 'g'),
          ('person', 'e'),
          ('parent', 'e', 'f'),
          ('parent', 'f', 'g')}


    X = [p1, n1, p2]
    y = [1, 0, 1]
    t = [('a', 'c'), ('a', 'g'), ('e', 'g')]

    learner = IncrementalHeuristic(args=('?A', '?B'),
                                   constraints=frozenset([('person', '?A'), 
                                                          ('person', '?B')]))

    for i, x in enumerate(X):
        print("Adding the following instance (%i):" % y[i])
        pprint(x)
        learner.ifit(t[i], x, y[i])
        print("Resulting hset")
        print(learner.get_hset())
        print(len(learner.get_hset()))
 
