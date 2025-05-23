using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Italbytz.ML.ModelBuilder.Configuration;
using Italbytz.ML.Tests.Util.ML;
using Italbytz.ML.UCIMLR;
using JetBrains.Annotations;
using Microsoft.ML;

namespace Italbytz.ML.Tests.Util;

public class DataHelper
{
    public static string GenerateModelBuilderConfig(ScenarioType scenario, IDataSource dataSource, 
        ITrainingOption trainingOption,[CanBeNull] IEnvironment environment = null)
    {
        environment ??= new LocalEnvironmentV1
        {
            Type = "LocalCPU",
            EnvironmentType = EnvironmentType.LocalCPU
        };
        var config = new TrainingConfiguration
        {
            Scenario = scenario,
            DataSource = dataSource,
            Environment = environment,
            TrainingOption = trainingOption
        };
        return config.SerializeToJson();
    }

    public static string GenerateModelBuilderConfigForDataset(Dataset dataSet,string trainingFilePath,
        ScenarioType scenario, 
        ITrainingOption trainingOption,[CanBeNull] IEnvironment environment = null)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters =
            {
                new JsonStringEnumConverter()
            }
        };
        var jsonColumnProperties = dataSet switch
        {
            Dataset.BalanceScale => ColumnPropertiesHelper.BalanceScale,
            Dataset.HeartDisease => ColumnPropertiesHelper.HeartDisease,
            Dataset.Iris => ColumnPropertiesHelper.Iris,
            Dataset.WineQuality => ColumnPropertiesHelper.WineQuality,
            Dataset.BreastCancerWisconsinDiagnostic => ColumnPropertiesHelper
                .BreastCancerWisconsinDiagnostic,
            _ => throw new ArgumentOutOfRangeException(nameof(dataSet), dataSet,
                null)
        };

        var columnProperties =
            JsonSerializer.Deserialize<ColumnPropertiesV5[]>(
                jsonColumnProperties, options);
        var dataSource = new TabularFileDataSourceV3
        {
            EscapeCharacter = '\\',
            ReadMultiLines = false,
            AllowQuoting = false,
            FilePath = trainingFilePath,
            Delimiter = ",",
            DecimalMarker = '.',
            HasHeader = true,
            ColumnProperties = columnProperties
        };
        return GenerateModelBuilderConfig(scenario, dataSource, trainingOption, environment);
    }

    public static string GenerateModelBuilderConfigForDataset(Dataset dataSet,
        string trainingFilePath,
        ScenarioType scenario,
        string labelColumn, int trainingTime,
        string[] trainers,IValidationOption validationOption,
        [CanBeNull] IEnvironment environment = null)
    {
        var trainingOption = new ClassificationTrainingOptionV2
        {
            Subsampling = false,
            TrainingTime = trainingTime,
            LabelColumn = labelColumn,
            AvailableTrainers = trainers,
            ValidationOption = validationOption
        };   
        return GenerateModelBuilderConfigForDataset(dataSet, trainingFilePath, scenario, trainingOption, environment);
    }

    /*public static string GenerateModelBuilderConfig(Dataset dataSet,
        string filePath, string labelColumn, int trainingTime,
        string[] trainers)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters =
            {
                new JsonStringEnumConverter()
            }
        };
        var jsonColumnProperties = dataSet switch
        {
            Dataset.BalanceScale => ColumnPropertiesHelper.BalanceScale,
            Dataset.HeartDisease => ColumnPropertiesHelper.HeartDisease,
            Dataset.Iris => ColumnPropertiesHelper.Iris,
            Dataset.WineQuality => ColumnPropertiesHelper.WineQuality,
            Dataset.BreastCancerWisconsinDiagnostic => ColumnPropertiesHelper
                .BreastCancerWisconsinDiagnostic,
            _ => throw new ArgumentOutOfRangeException(nameof(dataSet), dataSet,
                null)
        };

        var columnProperties =
            JsonSerializer.Deserialize<ColumnPropertiesV5[]>(
                jsonColumnProperties, options);

        var dataSource = new TabularFileDataSourceV3
        {
            EscapeCharacter = '\\',
            ReadMultiLines = false,
            AllowQuoting = false,
            FilePath = filePath,
            Delimiter = ",",
            DecimalMarker = '.',
            HasHeader = true,
            ColumnProperties = columnProperties
        };
        var environment = new LocalEnvironmentV1
        {
            Type = "LocalCPU",
            EnvironmentType = EnvironmentType.LocalCPU
        };
        var trainingOption = new ClassificationTrainingOptionV2
        {
            Subsampling = false,
            TrainingTime = trainingTime,
            LabelColumn = labelColumn,
            AvailableTrainers = trainers,
            ValidationOption = new TrainValidationSplitOptionV0
            {
                SplitRatio = 0.1f
            }
        };
        var config = new TrainingConfiguration
        {
            Scenario = ScenarioType.Classification,
            DataSource = dataSource,
            Environment = environment,
            TrainingOption = trainingOption
        };
        return JsonSerializer.Serialize(config, options);
    }

    public static void MakeTrainTestSets(IDataView dataView, string path)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var cvResults = mlContext.Data.CrossValidationSplit(dataView);
        var index = 0;
        foreach (var fold in cvResults)
        {
            index++;
            var trainData = fold.TrainSet;
            var testData = fold.TestSet;
            var trainDataPath =
                $"{path}_train_{index}.csv";
            var testDataPath =
                $"{path}_test_{index}.csv";
            using (var trainDataStream = new FileStream(trainDataPath,
                       FileMode.Create, FileAccess.Write))
            {
                mlContext.Data.SaveAsText(trainData, trainDataStream, ',');
            }

            using (var testDataStream = new FileStream(testDataPath,
                       FileMode.Create, FileAccess.Write))
            {
                mlContext.Data.SaveAsText(testData, testDataStream, ',');
            }
        }
    }*/

    public static Dictionary<string, float> ParseMLRun(string filePath)
    {
        using var reader = new StreamReader(filePath);
        var bestMacroaccuracy = new Dictionary<string, float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine()?.TrimStart();
            if (line[0] != '|') continue;
            var elements = line.Split(' ',
                StringSplitOptions.RemoveEmptyEntries |
                StringSplitOptions.TrimEntries);
            var nextIsAccuracy = false;
            var currentAlgorithm = "";
            foreach (var element in elements)
            {
                if (element.Contains("|")) continue;
                var parsedValue = 0.0f;
                if (float.TryParse(element, out parsedValue))
                {
                    if (nextIsAccuracy)
                    {
                        nextIsAccuracy = false;
                        if (bestMacroaccuracy[currentAlgorithm] < parsedValue)
                            bestMacroaccuracy[currentAlgorithm] = parsedValue;
                    }

                    continue;
                }

                nextIsAccuracy = true;
                currentAlgorithm = element;
                if (!bestMacroaccuracy.ContainsKey(element))
                    bestMacroaccuracy[element] = 0.0f;
            }
        }

        return bestMacroaccuracy;
    }
}