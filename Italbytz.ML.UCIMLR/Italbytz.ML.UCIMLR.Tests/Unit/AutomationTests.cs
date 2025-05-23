using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Italbytz.ML.Tests.Util;
using Italbytz.ML.UCIMLR;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Italbytz.ML.Tests.Unit;

[TestClass]
public class AutomationTests
{
    [TestMethod]
    public void SimulateIris()
    {
        var metrics = Simulate(Dataset.Iris, "class",
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            new[] { 3, 7, 13, 42, 73, 99, 256, 1024 }, 2);
        Console.WriteLine(
            string.Join(',',
                metrics.Select(m =>
                    m.MacroAccuracy.ToString(CultureInfo.InvariantCulture))));
    }


    public IEnumerable<Metrics> Simulate(Dataset dataset, string labelColumn,
        string[] trainers,
        int[] seeds, int trainingTime)
    {
        var metrics = new List<Metrics>();
        var data = UCIMLR.Data.Load(dataset);
        var tmpDir = Path.GetTempPath();
        var files =
            data.GenerateTrainValidateTestCsvs(tmpDir, dataset.ToString(),
                seeds: seeds);
        foreach (var file in files)
        {
            // Configure
            var configPath = GetConfiguration(tmpDir, file, dataset,
                labelColumn, trainers, trainingTime);
            // Run AutoML
            RunAutoMLForConfig(tmpDir, configPath);
            // Validate
            var metric = ValidateModel(tmpDir, file, dataset, labelColumn);
            metrics.Add(metric);
        }

        return metrics;
    }

    private Metrics ValidateModel(string tmpDir,
        TrainValidateTestFileNames file, Dataset dataset, string labelColumn)
    {
        var testData = Path.Combine(tmpDir, file.TestFileName);
        var modelPath = Path.Combine(tmpDir,
            "config.mlnet");
        var mlContext = new MLContext();
        try
        {
            var mlModel = mlContext.Model.Load(modelPath, out _);
            var testDataView = dataset switch
            {
                Dataset.HeartDisease => mlContext.Data
                    .LoadFromTextFile<HeartDiseaseModelInput>(
                        testData,
                        ',', true),
                Dataset.Iris => mlContext.Data
                    .LoadFromTextFile<IrisModelInput>(
                        testData,
                        ',', true),
                Dataset.WineQuality => mlContext.Data
                    .LoadFromTextFile<WineQualityModelInput>(
                        testData,
                        ',', true),
                Dataset.BreastCancerWisconsinDiagnostic => mlContext
                    .Data
                    .LoadFromTextFile<
                        BreastCancerWisconsinDiagnosticModelInput>(
                        testData,
                        ',', true),
                _ => throw new ArgumentOutOfRangeException(nameof(dataset),
                    dataset,
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
        Dataset dataset, string labelColumn, string[] trainers,
        int trainingTime)
    {
        var trainingData = Path.Combine(dir, file.TrainFileName);
        var validationData = Path.Combine(dir, file.ValidateFileName);
        var validationOption = new FileValidationOptionV0
        {
            FilePath = validationData
        };
        var config = DataHelper.GenerateModelBuilderConfigForDataset(
            dataset, trainingData, ScenarioType.Classification, labelColumn,
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

    protected Metrics SimulateMLNet(Dataset dataSet,
        string trainingData,
        string testData,
        string labelColumn, int trainingTime,
        string[] trainers, bool isMulticlass)
    {
        // Configure a Model Builder configuration
        var config = "XYZ";
        //DataHelper.GenerateModelBuilderConfig(dataSet, trainingData,
        //    labelColumn, trainingTime, trainers);
        Assert.IsNotNull(config);
        // Save the configuration
        var modelFileName = trainingData
            .Substring(trainingData.LastIndexOf('/') + 1)
            .Replace("train.csv", "");
        modelFileName = $"{modelFileName}{trainers[0]}";
        var configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            $"{modelFileName}.mbconfig");
        File.WriteAllText(configPath, config);
        // Run AutoML
        RunAutoMLForConfig(AppDomain.CurrentDomain.BaseDirectory,
            modelFileName);
        CleanUp(AppDomain.CurrentDomain.BaseDirectory);
        var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            $"{modelFileName}.mlnet");
        var mlContext = new MLContext();
        try
        {
            var mlModel = mlContext.Model.Load(modelPath, out _);
            IDataView testDataView = null;
            /*var testDataView = dataSet switch
            {
                Dataset.HeartDisease => mlContext.Data
                    .LoadFromTextFile<HeartDiseaseModelInputOriginal>(
                        testData,
                        ',', true),
                Dataset.Iris => mlContext.Data
                    .LoadFromTextFile<IrisModelInput>(
                        testData,
                        ',', true),
                Dataset.WineQuality => mlContext.Data
                    .LoadFromTextFile<WineQualityModelInputOriginal>(
                        testData,
                        ',', true),
                Dataset.BreastCancerWisconsinDiagnostic => mlContext
                    .Data
                    .LoadFromTextFile<
                        BreastCancerWisconsinDiagnosticModelInput>(
                        testData,
                        ',', true),
                _ => throw new ArgumentOutOfRangeException(nameof(dataSet),
                    dataSet,
                    null)
            };*/
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
                        $"Neither binary nor multiclass metrics available for {trainingData} and trainer {trainers[0]}.");
                }
            }
        }
        catch (Exception e)
        {
            Console.WriteLine(
                $"Error loading model for data set {trainingData} and trainer {trainers[0]}.");
        }

        return new Metrics
        {
            IsBinaryClassification = !isMulticlass,
            IsMulticlassClassification = isMulticlass,
            MacroAccuracy = 0.0f,
            Accuracy = 0.0f
        };
    }
}