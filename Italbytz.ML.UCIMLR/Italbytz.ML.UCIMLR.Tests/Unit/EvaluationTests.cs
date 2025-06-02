using System.Collections.Generic;
using System.Reflection;
using Italbytz.ML.ModelBuilder.Configuration;
using Italbytz.ML.Trainers;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Italbytz.ML.UCIMLR.Tests.Unit;

[TestClass]
public class EvaluationTests
{
    [TestMethod]
    public void EvaluateBreastCancerWisconsinDiagnosticOneVersusAllFastTree()
    {
        var data = Data.BreastCancerWisconsinDiagnostic;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(
            mlContext.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options
                {
                    NumberOfLeaves = 5, MinimumExampleCountPerLeaf = 19,
                    NumberOfTrees = 4, MaximumBinCountPerFeature = 279,
                    FeatureFraction = 0.99999999,
                    LearningRate = 0.18965803293955688,
                    LabelColumnName = @"Diagnosis",
                    FeatureColumnName = @"Features", DiskTranspose = false
                }), @"Diagnosis");
        var metrics = Evaluate(data, trainer);
    }

    public void EvaluateBreastCancerWisconsinDiagnosticFastTree()
    {
        var data = Data.BreastCancerWisconsinDiagnostic;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainer = mlContext.BinaryClassification.Trainers.FastTree(
            new FastTreeBinaryTrainer.Options
            {
                NumberOfLeaves = 5, MinimumExampleCountPerLeaf = 19,
                NumberOfTrees = 4, MaximumBinCountPerFeature = 279,
                FeatureFraction = 0.99999999,
                LearningRate = 0.18965803293955688,
                LabelColumnName = @"Diagnosis",
                FeatureColumnName = @"Features", DiskTranspose = false
            });
        var metrics = Evaluate(data, trainer);
    }

    [TestMethod]
    public void EvaluateIrisFastTree()
    {
        var data = Data.Iris;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(
            mlContext.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options
                {
                    NumberOfLeaves = 4,
                    MinimumExampleCountPerLeaf = 20,
                    NumberOfTrees = 4,
                    MaximumBinCountPerFeature = 254,
                    FeatureFraction = 1,
                    LearningRate = 0.09999999999999998,
                    LabelColumnName = @"class",
                    FeatureColumnName = @"Features",
                    DiskTranspose = false
                }), @"class");
        var metrics = Evaluate(data, trainer);
    }

    [TestMethod]
    public void EvaluateIrisLbgfsMaximumEntrop()
    {
        var data = Data.Iris;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainer =
            mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                new LbfgsMaximumEntropyMulticlassTrainer.Options
                {
                    L1Regularization = 1F, L2Regularization = 1F,
                    LabelColumnName = @"class", FeatureColumnName = @"Features"
                });
        var metrics = Evaluate(data, trainer);
    }

    [TestMethod]
    public void EvaluateIrisLbgfsMaximumEntropy()
    {
        var data = Data.Iris;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainer =
            mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                new LbfgsMaximumEntropyMulticlassTrainer.Options
                {
                    L1Regularization = 1F, L2Regularization = 1F,
                    LabelColumnName = @"class", FeatureColumnName = @"Features"
                });
        var metrics = Evaluate(data, trainer);
    }

    [TestMethod]
    public void EvaluateIrisFastForest()
    {
        var data = Data.Iris;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainer =
            mlContext.MulticlassClassification.Trainers.OneVersusAll(
                mlContext.BinaryClassification.Trainers.FastForest(
                    new FastForestBinaryTrainer.Options
                    {
                        NumberOfTrees = 4, NumberOfLeaves = 4,
                        FeatureFraction = 1F, LabelColumnName = @"class",
                        FeatureColumnName = @"Features"
                    }), @"class");
        var metrics = Evaluate(data, trainer);
    }

    [TestMethod]
    public void EvaluateIrisSdca()
    {
        var data = Data.Iris;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainer =
            mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                new SdcaMaximumEntropyMulticlassTrainer.Options
                {
                    L1Regularization = 0.11220099F,
                    L2Regularization = 0.10640355F, LabelColumnName = @"class",
                    FeatureColumnName = @"Features"
                });
        var metrics = Evaluate(data, trainer);
    }

    private MulticlassClassificationMetrics Evaluate(IDataset data,
        IEstimator<ITransformer> trainer)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = data.BuildPipeline(mlContext,
            ScenarioType.Classification, trainer);
        var model = pipeline.Fit(data.DataView);
        var predictions = model.Transform(data.DataView);
        return mlContext.MulticlassClassification.Evaluate(predictions,
            data.LabelColumnName);
    }

    private IEnumerable<ICanSaveModel> RetrieveModelParameters(
        ICanSaveModel transformer)
    {
        var modelParameters = new List<ICanSaveModel>();
        if (transformer is IEnumerable<ITransformer> chain)
        {
            foreach (var chainItem in chain)
                modelParameters.AddRange(RetrieveModelParameters(chainItem));
        }
        else if (transformer is IPredictionTransformer<ICanSaveModel>
                 predictionTransformer)
        {
            var model = predictionTransformer.Model;
            modelParameters.Add(model);

            if (model is OneVersusAllModelParameters
                oneVersusAllModelParameters)
                modelParameters.Add(oneVersusAllModelParameters.ToPublic());

            var subModelParamsProp = model?.GetType()
                .GetProperty("SubModelParameters",
                    BindingFlags.Instance | BindingFlags.NonPublic |
                    BindingFlags.Public);
            if (subModelParamsProp != null)
            {
                var subModelParams =
                    subModelParamsProp.GetValue(model) as IEnumerable<object>;
                if (subModelParams != null)
                    foreach (var subModel in subModelParams)
                    {
                        modelParameters.Add(subModel as ICanSaveModel);
                        modelParameters.AddRange(
                            RetrieveModelParameters(subModel as ICanSaveModel));
                    }
            }
        }

        return modelParameters;
    }

    private void ExplainOld(ITransformer transformer)
    {
        if (transformer is not IEnumerable<ITransformer> chain) return;
        foreach (var predictionTransformerCandidate in chain)
            if (predictionTransformerCandidate is
                IPredictionTransformer<ICanSaveModel>)
                transformer = predictionTransformerCandidate;
        if (transformer is not IPredictionTransformer<ICanSaveModel>
            predictionTransformer) return;
        var model = predictionTransformer.Model;

        // Zugriff auf die interne Property "SubModelParameters" per Reflection
        var subModelParamsProp = model?.GetType()
            .GetProperty("SubModelParameters",
                BindingFlags.Instance | BindingFlags.NonPublic |
                BindingFlags.Public);
        if (subModelParamsProp != null)
        {
            var subModelParams =
                subModelParamsProp.GetValue(model) as IEnumerable<object>;
            if (subModelParams != null)
                foreach (var subModel in subModelParams)
                {
                    // Hier kann weitere Verarbeitung erfolgen, z\.B\. Logging oder Analyse
                }
        }


        /*if (model is TransformerChain<KeyToValueMappingTransformer> chain)
        {
            foreach (var transformer in chain)
            {
                if (transformer is MulticlassPredictionTransformer<
                        OneVersusAllModelParameters> ova)
                {
                    var ovaModel = ova.Model;
                    foreach (var submodel in ovaModel.SubModelParameters)
                    {

                    }

                }
            }
            return;
        }*/
    }
}