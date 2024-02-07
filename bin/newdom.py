import datetime
import os
import random

import binutil  # required to import from dreamcoder modules

from dreamcoder.dreamcoder import commandlineArguments, ecIterator, DummyFeatureExtractor
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

# Primitives
def _incr_custom(x): return lambda x: x + 1
def _incr2_custom(x): return lambda x: x + 2


def addN_custom(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}


def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )


if __name__ == "__main__":

    # Options more or less copied from list.py

    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh',
        iterations=10, recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs())

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/demo/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # Create list of primitives

    primitives = [
        Primitive("incr_custom", arrow(tint, tint), _incr_custom),
        Primitive("incr2_custom", arrow(tint, tint), _incr2_custom),
    ]

    # Create grammar

    grammar = Grammar.uniform(primitives)

    def add1_custom(): return addN_custom(1)
    def add2_custom(): return addN_custom(2)
    def add3_custom(): return addN_custom(3)

    # Training data

    training_examples = [
        {"name": "add1", "examples": [add1_custom() for _ in range(5000)]},
        {"name": "add2", "examples": [add2_custom() for _ in range(5000)]},
        {"name": "add3", "examples": [add3_custom() for _ in range(5000)]},
    ]
    training = [get_tint_task(item) for item in training_examples]

    # Testing data

    def add4(): return addN_custom(4)
    def add0(): return addN_custom(0)

    testing_examples = [
        {"name": "add4", "examples": [add4() for _ in range(500)]},
        {"name": "add6", "examples": [add0() for _ in range(500)]}
    ]
    testing = [get_tint_task(item) for item in testing_examples]

    # EC iterate

    # commandlineArguments(
    #     featureExtractor=DummyFeatureExtractor,
    #     iterations=6,
    #     CPUs=numberOfCPUs(),
    #     structurePenalty=1.,
    #     recognitionTimeout=7200,
    #     helmholtzRatio=0.5,
    #     activation="tanh",
    #     maximumFrontier=5,
    #     a=3,
    #     topK=2,
    #     pseudoCounts=30.0,
    #     extras=rational_options)
    generator = ecIterator(grammar,
                           training,
                           testingTasks=testing,
                           featureExtractor=DummyFeatureExtractor,
                           enumerationTimeout=2,
                           testingTimeout=2,
                           iterations=5,
                           recognitionEpochs=[1000],
                           parser="loglinear",
                           maximumFrontier=5,
                           recognitionTimeout=100)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))