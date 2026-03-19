using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Italbytz.ML.Data.Tests.Unit;

[TestClass]
public class PipelineTests
{
    [TestMethod]
    public void TestCarEvaluationPipeline()
    {
        var dataset = Data.CarEvaluation;
        var data = dataset.DataView;

        var mlContext = ThreadSafeMLContext.LocalMLContext;

        var trainer = mlContext.MulticlassClassification.Trainers
            .SdcaMaximumEntropy();
        var pipeline = dataset.BuildPipeline(mlContext, trainer);


        var model = pipeline.Fit(data);
        var predictions = model.Transform(data);

        var metrics = mlContext.MulticlassClassification
            .Evaluate(predictions);

        Assert.IsTrue(metrics.MacroAccuracy > 0.7);
    }

    [TestMethod]
    public void TestObesityLevelsPipeline()
    {
        var dataset = Data.ObesityLevels;
        var data = dataset.DataView;

        var mlContext = ThreadSafeMLContext.LocalMLContext;

        var trainer = mlContext.MulticlassClassification.Trainers
            .SdcaMaximumEntropy();
        var pipeline = dataset.BuildPipeline(mlContext, trainer);


        var model = pipeline.Fit(data);
        var predictions = model.Transform(data);

        var metrics = mlContext.MulticlassClassification
            .Evaluate(predictions);

        Assert.IsTrue(metrics.MacroAccuracy > 0.7);
    }
}