using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Italbytz.ML.Tests.Util;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Italbytz.ML.UCIMLR.Tests.Unit;

[TestClass]
public class AutomationTests
{
    [TestMethod]
    public void SimulateIris()
    {
        var data = Data.Iris;
        var metrics = Simulate(data, "class",
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            new[] { 3, 7, 13, 42, 73, 99, 256, 1024 }, 2);
        Console.WriteLine(
            string.Join(',',
                metrics.Select(m =>
                    m.MacroAccuracy.ToString(CultureInfo.InvariantCulture))));
    }


    public IEnumerable<Metrics> Simulate(IDataset dataset,
        string labelColumn,
        string[] trainers,
        int[] seeds, int trainingTime)
    {
        var metrics = new List<Metrics>();

        // ToDo: Refactor
        var datasetEnum = DatasetEnum.WineQuality;
        if (dataset is IrisDataset)
            datasetEnum = DatasetEnum.Iris;

        var tmpDir = Path.GetTempPath();
        var files =
            dataset.GetTrainValidateTestFiles(tmpDir, datasetEnum.ToString(),
                seeds: seeds);
        foreach (var file in files)
        {
            // Configure
            var configPath = GetConfiguration(tmpDir, file, datasetEnum,
                labelColumn, trainers, trainingTime);
            // Run AutoML
            RunAutoMLForConfig(tmpDir, configPath);
            // Validate
            var metric = ValidateModel(tmpDir, file, datasetEnum, labelColumn);
            metrics.Add(metric);
        }

        return metrics;
    }

    private Metrics ValidateModel(string tmpDir,
        TrainValidateTestFileNames file, DatasetEnum datasetEnum,
        string labelColumn)
    {
        var testData = Path.Combine(tmpDir, file.TestFileName);
        var modelPath = Path.Combine(tmpDir,
            "config.mlnet");
        var mlContext = new MLContext();
        try
        {
            var mlModel = mlContext.Model.Load(modelPath, out _);
            var testDataView = datasetEnum switch
            {
                DatasetEnum.HeartDisease => mlContext.Data
                    .LoadFromTextFile<HeartDiseaseModelInput>(
                        testData,
                        ',', true),
                DatasetEnum.Iris => mlContext.Data
                    .LoadFromTextFile<IrisModelInput>(
                        testData,
                        ',', true),
                DatasetEnum.WineQuality => mlContext.Data
                    .LoadFromTextFile<WineQualityModelInput>(
                        testData,
                        ',', true),
                DatasetEnum.BreastCancerWisconsinDiagnostic => mlContext
                    .Data
                    .LoadFromTextFile<
                        BreastCancerWisconsinDiagnosticModelInput>(
                        testData,
                        ',', true),
                _ => throw new ArgumentOutOfRangeException(nameof(datasetEnum),
                    datasetEnum,
                    null)
            };
            var testResult = mlModel.Transform(testDataView);
            try
            {
                var metrics = mlContext.BinaryClassification
                    .Evaluate(testResult, labelColumn);
                return new Metrics
                {
                    IsBinaryClassification = true,
                    Accuracy = metrics.Accuracy,
                    AreaUnderRocCurve = metrics.AreaUnderRocCurve,
                    F1Score = metrics.F1Score,
                    AreaUnderPrecisionRecallCurve =
                        metrics.AreaUnderPrecisionRecallCurve
                };
            }
            catch (Exception e1)
            {
                try
                {
                    var metrics = mlContext.MulticlassClassification
                        .Evaluate(testResult, labelColumn);
                    return new Metrics
                    {
                        IsMulticlassClassification = true,
                        MacroAccuracy = metrics.MacroAccuracy
                    };
                }
                catch (Exception e2)
                {
                    Console.WriteLine(
                        $"Neither binary nor multiclass metrics available for {testData}.");
                }
            }
        }
        catch (Exception e)
        {
            Console.WriteLine(
                $"Error loading model for data set {testData}.");
        }

        return new Metrics
        {
            MacroAccuracy = 0.0f,
            Accuracy = 0.0f
        };
    }

    private string GetConfiguration(string dir, TrainValidateTestFileNames file,
        DatasetEnum datasetEnum, string labelColumn, string[] trainers,
        int trainingTime)
    {
        var trainingData = Path.Combine(dir, file.TrainFileName);
        var validationData = Path.Combine(dir, file.ValidateFileName);
        var validationOption = new FileValidationOptionV0
        {
            FilePath = validationData
        };
        var config = DataHelper.GenerateModelBuilderConfigForDataset(
            datasetEnum, trainingData, ScenarioType.Classification, labelColumn,
            trainingTime,
            trainers,
            validationOption);
        var configPath = Path.Combine(dir,
            "config.mbconfig");
        File.WriteAllText(configPath, config);
        return configPath;
    }

    private void RunAutoMLForConfig(string workingDirectory, string configFile)
    {
        var mlnet = new Process();
        mlnet.StartInfo.FileName = "mlnet";
        mlnet.StartInfo.WorkingDirectory =
            workingDirectory;
        mlnet.StartInfo.Arguments =
            $"train --training-config {configFile} -v q";
        mlnet.Start();
        mlnet.WaitForExit();
    }

    private void CleanUp(string directory)
    {
        var projectFiles = Directory.GetFiles(directory, "*.cs");
        foreach (var file in projectFiles) File.Delete(file);
        projectFiles = Directory.GetFiles(directory, "*.csproj");
        foreach (var file in projectFiles) File.Delete(file);
    }
}