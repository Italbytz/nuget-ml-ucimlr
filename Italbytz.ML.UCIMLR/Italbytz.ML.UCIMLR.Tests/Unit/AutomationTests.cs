using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using Italbytz.ML.ModelBuilder.Configuration;
using Italbytz.ML.Tests.Util;
using Italbytz.ML.Trainers;
using Italbytz.ML.Trainers.FastTree;
using JetBrains.Annotations;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Italbytz.ML.Data.Tests.Unit;

[TestClass]
public class AutomationTests
{
    private readonly int[] _seeds =
    [
        42, 7, 13, 99, 256, 1024, 73, 3, 17, 23,
        5, 11, 19, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 79, 83, 89, 97, 101, 103,
        107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
        163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
        223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
        271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
        337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
        397, 401, 409, 419, 421, 431, 433, 439, 443, 449,
        457, 461, 463, 467, 479, 487, 491, 499, 503, 509
    ];

    private string _timeStamp;

    private StreamWriter LogWriter { get; set; }

    [TestMethod]
    public void SimulateIrisClassification()
    {
        var data = Data.Iris;
        var metrics = Simulate(data, ScenarioType.Classification,
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            _seeds, 60, 0.2f);
        var accuracies = metrics.Select(m =>
            m.MacroAccuracy.ToString(CultureInfo.InvariantCulture));
        File.WriteAllLines(
            "/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/Iris/AutoML.csv",
            accuracies);
        Console.WriteLine(
            string.Join(',', accuracies));
    }

    [TestMethod]
    public void ExplainIrisClassification()
    {
        var data = Data.Iris;
        Simulate(data, ScenarioType.Classification,
            ["FASTTREE"],
            _seeds, 60, 0.2f);
        LogWriter.Close();
    }

    [TestMethod]
    public void SimulateHeartDiseaseClassification()
    {
        var data = Data.HeartDisease;
        var metrics = Simulate(data, ScenarioType.Classification,
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            _seeds, 60, 0.2f);
        var accuracies = metrics.Select(m =>
            m.MacroAccuracy.ToString(CultureInfo.InvariantCulture));
        LogWriter.Close();
        File.WriteAllLines(
            "/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/HeartDisease/AutoML.csv",
            accuracies);
        Console.WriteLine(
            string.Join(',', accuracies));
    }

    [TestMethod]
    public void ExplainHeartDiseaseBinaryClassification()
    {
        var data = Data.HeartDiseaseBinary;
        Simulate(data, ScenarioType.Classification,
            ["FASTTREE"],
            _seeds, 60, 0.2f, true);
        LogWriter.Close();
    }

    [TestMethod]
    public void SimulateHeartDiseaseBinaryClassification()
    {
        var data = Data.HeartDiseaseBinary;
        var metrics = Simulate(data, ScenarioType.Classification,
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            _seeds, 60, 0.2f, true);
        var accuracies = metrics.Select(m =>
            m.F1Score.ToString(CultureInfo.InvariantCulture));
        LogWriter.Close();
        File.WriteAllLines(
            "/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/HeartDisease/AutoMLBinary.csv",
            accuracies);
        Console.WriteLine(
            string.Join(',', accuracies));
    }

    [TestMethod]
    public void SimulateBreastCancerWisconsinDiagnosticBinaryClassification()
    {
        var data = Data.BreastCancerWisconsinDiagnostic;
        var metrics = Simulate(data, ScenarioType.Classification,
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            _seeds, 60, 0.2f, true);
        var accuracies = metrics.Select(m =>
            m.F1Score.ToString(CultureInfo.InvariantCulture));
        LogWriter.Close();
        File.WriteAllLines(
            "/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/BreastCancerWisconsinDiagnostic/AutoMLBinary.csv",
            accuracies);
        Console.WriteLine(
            string.Join(',', accuracies));
    }

    [TestMethod]
    public void SimulateWineQualityClassification()
    {
        var data = Data.WineQuality;
        var metrics = Simulate(data, ScenarioType.Classification,
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            _seeds, 60, 0.2f);
        var rSquared = metrics.Select(m =>
            m.MacroAccuracy.ToString(CultureInfo.InvariantCulture));
        LogWriter.Close();
        File.WriteAllLines(
            "/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/WineQuality/AutoML.csv",
            rSquared);
        Console.WriteLine(
            string.Join(',', rSquared));
    }

    [TestMethod]
    public void SimulateWineQualityRegression()
    {
        var data = Data.WineQuality;
        var metrics = Simulate(data, ScenarioType.Regression,
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            _seeds, 60, 0.2f);
        var accuracies = metrics.Select(m =>
            m.RSquared.ToString(CultureInfo.InvariantCulture));
        LogWriter.Close();
        File.WriteAllLines(
            "/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/WineQuality/AutoMLRegression.csv",
            accuracies);
        Console.WriteLine(
            string.Join(',', accuracies));
    }

    [TestMethod]
    public void SimulateBreastCancerWisconsinDiagnosticClassification()
    {
        var data = Data.BreastCancerWisconsinDiagnostic;
        var metrics = Simulate(data, ScenarioType.Classification,
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            [3, 7, 13, 42, 73, 99, 256, 1024], 2, 0.2f);
        var accuracies = metrics.Select(m =>
            m.MacroAccuracy.ToString(CultureInfo.InvariantCulture));
        Console.WriteLine(
            string.Join(',', accuracies));
    }

    [TestMethod]
    public void SimulateAdultBinaryClassification()
    {
        var data = Data.Adult;
        var metrics = Simulate(data, ScenarioType.Classification,
            ["LBFGS", "FASTFOREST", "SDCA", "FASTTREE"],
            _seeds, 60, 0.2f, true);
        var accuracies = metrics.Select(m =>
            m.MacroAccuracy.ToString(CultureInfo.InvariantCulture));
        LogWriter.Close();
        File.WriteAllLines(
            "/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/Adult/AutoMLBinary.csv",
            accuracies);
        Console.WriteLine(
            string.Join(',', accuracies));
    }

    private IEnumerable<Metrics> Simulate(IDataset dataset,
        ScenarioType scenario, string[] trainers,
        int[] seeds, int trainingTime, float splitRatio, bool binary = false)
    {
        var metrics = new List<Metrics>();

        var tmpDir = Path.GetTempPath();
        _timeStamp = DateTime.Now.ToString("yyyyMMddHHmmss");
        var logPath = Path.Combine(tmpDir,
            $"{dataset.FilePrefix}_{_timeStamp}_Metrics.csv");
        LogWriter = new StreamWriter(logPath);
        LogWriter.WriteLine(
            "\"x\"");

        var files =
            dataset.GetTrainValidateTestFiles(tmpDir,
                validateFraction: splitRatio, testFraction: splitRatio,
                seeds: seeds);
        foreach (var file in files)
        {
            // Configure
            var configPath = GetConfigurationFileTrainValidationSplit(tmpDir,
                file, scenario,
                dataset,
                trainers, trainingTime, splitRatio);
            // Run AutoML
            RunAutoMLForConfig(tmpDir, configPath);
            // Validate
            var metric = ValidateModel(tmpDir, file, dataset, binary);
            metrics.Add(metric);
            var metricForCSV = scenario switch
            {
                ScenarioType.Classification => metric.IsBinaryClassification
                    ? metric.F1Score
                    : metric.MacroAccuracy,
                ScenarioType.Regression => metric.RSquared,
                _ => throw new ArgumentOutOfRangeException(nameof(scenario),
                    scenario, null)
            };
            LogWriter?.WriteLine(
                $"{metricForCSV}");

            LogWriter?.Flush();
        }

        return metrics;
    }

    [CanBeNull]
    private Metrics ValidateModel(string tmpDir,
        TrainValidateTestFileNames file, IDataset dataset, bool binary)
    {
        var testData = Path.Combine(tmpDir, file.TestFileName);
        var modelPath = Path.Combine(tmpDir,
            "config.mlnet");
        var mlContext = new MLContext();
        Metrics? metrics = null;
        try
        {
            var mlModel = mlContext.Model.Load(modelPath, out _);
            var testDataView = dataset.LoadFromTextFile(testData,
                ',', true);
            var testResult = mlModel.Transform(testDataView);
            ConfusionMatrix? confusionMatrix = null;
            string? pfiTable = null;
            var modelParameters = mlModel.GetModelParameters();
            try
            {
                var binaryMetrics = mlContext.BinaryClassification
                    .Evaluate(testResult, dataset.LabelColumnName);
                confusionMatrix = binaryMetrics.ConfusionMatrix;
                metrics = new Metrics
                {
                    IsBinaryClassification = true,
                    Accuracy = binaryMetrics.Accuracy,
                    AreaUnderRocCurve = binaryMetrics.AreaUnderRocCurve,
                    F1Score = binaryMetrics.F1Score,
                    AreaUnderPrecisionRecallCurve =
                        binaryMetrics.AreaUnderPrecisionRecallCurve
                };
            }
            catch (Exception e1)
            {
                try
                {
                    var multiclassMetrics = mlContext.MulticlassClassification
                        .Evaluate(testResult, dataset.LabelColumnName);
                    confusionMatrix = multiclassMetrics.ConfusionMatrix;
                    pfiTable =
                        mlContext.MulticlassClassification
                            .GetPermutationFeatureImportanceTable(mlModel,
                                testResult, dataset.LabelColumnName,
                                Metric.MacroAccuracy);
                    if (binary)
                        metrics = new Metrics
                        {
                            IsBinaryClassification = true,
                            Accuracy = multiclassMetrics.MicroAccuracy,
                            MacroAccuracy = multiclassMetrics.MacroAccuracy,
                            F1Score = multiclassMetrics.F1ScoreBinary()
                        };

                    metrics = new Metrics
                    {
                        IsMulticlassClassification = true,
                        Accuracy = multiclassMetrics.MicroAccuracy,
                        MacroAccuracy = multiclassMetrics.MacroAccuracy
                    };
                }
                catch (Exception e2)
                {
                    try
                    {
                        var regressionMetrics = mlContext.Regression
                            .Evaluate(testResult, dataset.LabelColumnName);
                        metrics = new Metrics
                        {
                            IsRegression = true,
                            RSquared = regressionMetrics.RSquared,
                            MeanAbsoluteError =
                                regressionMetrics.MeanAbsoluteError,
                            MeanSquaredError =
                                regressionMetrics.MeanSquaredError,
                            RootMeanSquaredError =
                                regressionMetrics.RootMeanSquaredError
                        };
                    }
                    catch (Exception e3)
                    {
                        Console.WriteLine(
                            $"No binary, multiclass or regression metrics available for {testData}.");
                    }
                }
            }

            ExplainModel(tmpDir, dataset.FilePrefix, modelParameters,
                confusionMatrix, pfiTable);
        }
        catch (Exception e)
        {
            Console.WriteLine(
                $"Error loading model for data set {testData}.");
        }


        return metrics;
    }

    private void ExplainModel(string tmpDir, string filePrefix,
        ICanSaveModel modelParameters, ConfusionMatrix? confusionMatrix,
        string? pfiTable)
    {
        _timeStamp = DateTime.Now.ToString("yyyyMMddHHmmss");
        var explanationsDir = Path.Combine(tmpDir, "explanations");
        if (!Directory.Exists(explanationsDir))
            Directory.CreateDirectory(explanationsDir);

        // Confusion matrix
        if (confusionMatrix != null)
        {
            var confusionMatrixPath = Path.Combine(explanationsDir,
                $"{filePrefix}_{_timeStamp}_ConfusionMatrix.");
            File.WriteAllText(confusionMatrixPath + "txt",
                confusionMatrix.GetFormattedConfusionTable());
            File.WriteAllText(confusionMatrixPath + "py",
                confusionMatrix.SklearnScript(confusionMatrixPath + "pdf"));
        }

        // PFI
        var pfiPath = Path.Combine(explanationsDir,
            $"{filePrefix}_{_timeStamp}_PFI.csv");
        if (pfiTable == null)
        {
            pfiTable = "No PFI table available.";
        }
        else
        {
            pfiTable = pfiTable.Replace("\r\n", "\n");
            pfiTable = pfiTable.Replace("\r", "\n");
        }

        File.WriteAllText(pfiPath, pfiTable);

        // Model parameters
        // ToDO: Currently hard coded for FastTreeBinaryModelParameters
        if (modelParameters is PublicOneVersusAllModelParameters pova)
            foreach (var submodel in pova.SubModelParameters)
                if (submodel is CalibratedModelParametersBase cmp)
                {
                    var subModelProperty =
                        typeof(CalibratedModelParametersBase).GetProperty(
                            "SubModel",
                            BindingFlags.DeclaredOnly | BindingFlags.Instance |
                            BindingFlags.NonPublic |
                            BindingFlags.Public | BindingFlags.Static);
                    if (subModelProperty != null)
                    {
                        var subModels = subModelProperty.GetValue(submodel);
                        if (subModels is FastTreeBinaryModelParameters
                            treeBinaryModelParameters)
                        {
                            var trees = treeBinaryModelParameters
                                .TrainedTreeEnsemble;
                            var index = 0;
                            foreach (var tree in trees.Trees)
                            {
                                var graphviz = tree.ToGraphviz();
                                var graphvizPath = Path.Combine(
                                    explanationsDir,
                                    $"{filePrefix}_{_timeStamp}_Tree_{index}.dot");
                                File.WriteAllText(graphvizPath, graphviz);
                                var plantuml = tree.ToPlantUML();
                                var plantumlPath = Path.Combine(
                                    explanationsDir,
                                    $"{filePrefix}_{_timeStamp}_Tree_{index}.pu");
                                File.WriteAllText(plantumlPath, plantuml);

                                index++;
                            }
                        }
                    }
                }
        /*                if (subModelsProperty != null)
                {
                    var subModels = subModelsProperty.GetValue(submodel);
                    if (subModels != null)
                        if (subModels is FastTreeBinaryModelParameters
                            treeParameters)
                        {
                            var ensemble = treeParameters.TrainedTreeEnsemble;
                            var index = 0;
                            foreach (var tree in ensemble.Trees)
                            {
                                var graphviz = tree.ToGraphviz();
                                var graphvizPath = Path.Combine(
                                    explanationsDir,
                                    $"{filePrefix}_{_timeStamp}_Tree_{index}.dot");
                                File.WriteAllText(graphvizPath, graphviz);
                                var plantuml = tree.ToPlantUML();
                                var plantumlPath = Path.Combine(
                                    explanationsDir,
                                    $"{filePrefix}_{_timeStamp}_Tree_{index}.pu");
                                File.WriteAllText(plantumlPath, plantuml);
                                index++;
                            }
                        }
                }*/
    }

    private string GetConfiguration(string dir, string trainingData,
        ScenarioType scenario,
        IDataset dataset, string[] trainers,
        int trainingTime, IValidationOption validationOption)
    {
        var config = DataHelper.GenerateModelBuilderConfigForDataset(
            dataset, trainingData, scenario,
            dataset.LabelColumnName,
            trainingTime,
            trainers,
            validationOption);
        var configPath = Path.Combine(dir,
            "config.mbconfig");
        File.WriteAllText(configPath, config);
        return configPath;
    }

    private string GetConfigurationFileTrainValidationSplit(string dir,
        TrainValidateTestFileNames file, ScenarioType scenario,
        IDataset dataset, string[] trainers,
        int trainingTime, float splitRatio = 0.2f)
    {
        var trainingData = Path.Combine(dir, file.TrainValidateFileName);
        var validationOption = new TrainValidationSplitOptionV0
        {
            SplitRatio = splitRatio
        };
        return GetConfiguration(dir, trainingData, scenario, dataset, trainers,
            trainingTime, validationOption);
    }

    private string GetConfigurationFileValidation(string dir,
        TrainValidateTestFileNames file, ScenarioType scenario,
        IDataset dataset, string[] trainers,
        int trainingTime)
    {
        var trainingData = Path.Combine(dir, file.TrainFileName);
        var validationData = Path.Combine(dir, file.ValidateFileName);
        var validationOption = new FileValidationOptionV0
        {
            FilePath = validationData
        };
        return GetConfiguration(dir, trainingData, scenario, dataset, trainers,
            trainingTime, validationOption);
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