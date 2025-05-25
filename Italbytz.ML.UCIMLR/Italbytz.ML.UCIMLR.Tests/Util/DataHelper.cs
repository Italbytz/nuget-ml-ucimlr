using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Italbytz.ML.ModelBuilder.Configuration;
using Italbytz.ML.Tests.Util.ML;
using Italbytz.ML.UCIMLR;
using JetBrains.Annotations;

namespace Italbytz.ML.Tests.Util;

public class DataHelper
{
    public static string GenerateModelBuilderConfig(ScenarioType scenario,
        IDataSource dataSource,
        ITrainingOption trainingOption,
        [CanBeNull] IEnvironment environment = null)
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
        return config.SerializeToJson(true);
    }

    public static string GenerateModelBuilderConfigForDataset(
        DatasetEnum dataSet,
        string trainingFilePath,
        ScenarioType scenario,
        ITrainingOption trainingOption,
        [CanBeNull] IEnvironment environment = null)
    {
        var options = new JsonSerializerOptions
        {
            Converters =
            {
                new JsonStringEnumConverter()
            }
        };
        var jsonColumnProperties = dataSet switch
        {
            DatasetEnum.BalanceScale => ColumnPropertiesHelper.BalanceScale,
            DatasetEnum.HeartDisease => ColumnPropertiesHelper.HeartDisease,
            DatasetEnum.Iris => ColumnPropertiesHelper.Iris,
            DatasetEnum.WineQuality => ColumnPropertiesHelper.WineQuality,
            DatasetEnum.BreastCancerWisconsinDiagnostic =>
                ColumnPropertiesHelper
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
        return GenerateModelBuilderConfig(scenario, dataSource, trainingOption,
            environment);
    }

    public static string GenerateModelBuilderConfigForDataset(
        DatasetEnum dataSet,
        string trainingFilePath,
        ScenarioType scenario,
        string labelColumn, int trainingTime,
        string[] trainers, IValidationOption validationOption,
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
        return GenerateModelBuilderConfigForDataset(dataSet, trainingFilePath,
            scenario, trainingOption, environment);
    }

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