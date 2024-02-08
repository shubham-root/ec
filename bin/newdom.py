import datetime
import os
import random

import binutil  # required to import from dreamcoder modules

from dreamcoder.dreamcoder import commandlineArguments, ecIterator, DummyFeatureExtractor, RecurrentFeatureExtractor
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

# Primitives
def _incr_custom(x): return lambda x: x + 1
def _incr2_custom(x): return lambda x: x + 2

def _mult_custom(x): return lambda x: x * 1
def _mult2_custom(x): return lambda x: x * 2
def _mult3_custom(x): return lambda x: x * 3

def addN_custom(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}

def multN_custom(n):
    x = random.choice(range(500))
    return {"i":x, "o": x * n}

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
        Primitive("mult_custom", arrow(tint, tint), _mult_custom),
        Primitive("mult2_custom", arrow(tint, tint), _mult2_custom),
        Primitive("mult3_custom", arrow(tint, tint), _mult3_custom)
    ]

    # Create grammar

    grammar = Grammar.uniform(primitives)

    def add1_custom(): return addN_custom(1)
    def add2_custom(): return addN_custom(2)
    def add3_custom(): return addN_custom(3)
    def mult1_custom(): return multN_custom(1)
    def mult2_custom(): return multN_custom(2)
    def mult3_custom(): return multN_custom(3)

    # Training data

    training_examples = [
        {"name": "add1_custom", "examples": [add1_custom() for _ in range(50)]},
        {"name": "add2_custom", "examples": [add2_custom() for _ in range(50)]},
        {"name": "add3_custom", "examples": [add3_custom() for _ in range(50)]},
        {"name": "mult1_custom", "examples": [mult1_custom() for _ in range(50)]},
        {"name": "mult2_custom", "examples": [mult2_custom() for _ in range(50)]},
        {"name": "mult3_custom", "examples": [mult3_custom() for _ in range(50)]},
    ]
    training = [get_tint_task(item) for item in training_examples]
    print("🚀 ~ training:", training)

    # Testing data

    def add4(): return addN_custom(4)
    def add0(): return addN_custom(0)
    
    def mul7(): return multN_custom(7)
    def mul6(): return multN_custom(6)
    
    # def d_3m_3a_2a(): return mult3_custom(addN_custom(addN_custom(2)))
    

    testing_examples = [
        {"name": "add4_custom", "examples": [add4() for _ in range(50)]},
        {"name": "add0_custom", "examples": [add0() for _ in range(50)]},
        {"name": "mult7_custom", "examples": [mul7() for _ in range(50)]},
        {"name": "mult6_custom", "examples": [mul6() for _ in range(50)]},
        # {"name": "d_3m_3a_2a_custom", "examples": [d_3m_3a_2a() for _ in range(500)]},
    ]
    testing = [get_tint_task(item) for item in testing_examples]
    print("🚀 ~ testing:", testing)

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
                           featureExtractor=RecurrentFeatureExtractor,
                           enumerationTimeout=2,
                           testingTimeout=2,
                           iterations=5,
                           recognitionEpochs=[1000],
                           parser="loglinear",
                           maximumFrontier=5,
                           recognitionTimeout=100,
                           compressor='stitch',
                           useDSL=True,
                           testEvery=1)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))