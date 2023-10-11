import math
import openturns as ot
import openturns.viewer as otv
import openturns.viewer as viewer
import pandas as pd
import borehole_function
from operator import itemgetter
from matplotlib import pylab as plt
ot.Log.Show(ot.Log.NONE)


im = borehole_function.BoreholeModel()
sampleSize = 1000
inputTrain = im.distributionX.getSample(sampleSize)
outputTrain = im.model(inputTrain)

# sampleSize = 1000
inputSample = im.distributionX.getSample(sampleSize)
outputSample = im.model(inputTrain)



chaosalgo = ot.FunctionalChaosAlgorithm(inputTrain, outputTrain)

multivariateBasis = ot.OrthogonalProductPolynomialFactory([im.rw, im.r, im.Tu, im.Hu, im.Tl, im.Hl, im.L, im.Kw])

selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
projectionStrategy = ot.LeastSquaresStrategy(selectionAlgorithm)
totalDegree = 8
enumerateFunction = multivariateBasis.getEnumerateFunction()
basisSize = enumerateFunction.getBasisSizeFromTotalDegree(totalDegree)
adaptiveStrategy = ot.FixedStrategy(multivariateBasis, basisSize)
chaosAlgo = ot.FunctionalChaosAlgorithm(
    inputTrain, outputTrain, im.distributionX, adaptiveStrategy, projectionStrategy
)

chaosAlgo.run()
chaosResult = chaosAlgo.getResult()


metamodel = chaosResult.getMetaModel()

n_valid = 1000
inputTest = im.distributionX.getSample(n_valid)
outputTest = im.model(inputTest)
val = ot.MetaModelValidation(inputTest, outputTest, metamodel)
Q2 = val.computePredictivityFactor()[0]
# Q2

graph = val.drawValidation()
graph.setTitle("Q2=%.2f%%" % (Q2 * 100))
view = otv.View(graph)


plt.plot(outputSample.sort(), label='Simulation (Real)')
plt.plot(chaosResult.getOutputSample().sort(), label='Approximation')
plt.legend()
plt.xlabel("samples")
plt.ylabel("y0, Flow reate")


def find_best_parameters(input_data, output_data):
    in_data = input_data.asDataFrame()
    out_data = output_data.asDataFrame()
    max_out= out_data.idxmax().values
    best_in=in_data.iloc[max_out]
    result_best=pd.DataFrame()
    result_best[list(best_in.columns)]=best_in.values
    result_best['Name']='best'
    result_best['MaxFlow']= max_out
    result_best=result_best.set_index('Name')
    final= result_best.to_dict()
    print(result_best.T)


chaosSI = ot.FunctionalChaosSobolIndices(chaosResult)

find_best_parameters(inputSample, outputSample)

find_best_parameters(chaosResult.getInputSample(), chaosResult.getOutputSample())

dim_input = im.distributionX.getDimension()
first_order = [chaosSI.getSobolIndex(i) for i in range(dim_input)]
total_order = [chaosSI.getSobolTotalIndex(i) for i in range(dim_input)]
input_names = im.model.getInputDescription()
graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(input_names, first_order, total_order)
view = otv.View(graph)