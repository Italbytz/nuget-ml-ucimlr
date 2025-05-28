using Italbytz.ML.ModelBuilder.Configuration;
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
}