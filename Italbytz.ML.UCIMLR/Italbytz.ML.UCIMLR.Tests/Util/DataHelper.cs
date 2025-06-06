using System;
using System.Collections.Generic;
using System.IO;
using Italbytz.ML.ModelBuilder.Configuration;
using Italbytz.ML.Data;
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
        IDataset dataSet,
        string trainingFilePath,
        ScenarioType scenario,
        ITrainingOption trainingOption,
        [CanBeNull] IEnvironment environment = null)
    {
        var columnProperties =
            dataSet.ColumnProperties;
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
        IDataset dataSet,
        string trainingFilePath,
        ScenarioType scenario,
        string labelColumn, int trainingTime,
        string[] trainers, IValidationOption validationOption,
        [CanBeNull] IEnvironment environment = null)
    {
        ITrainingOption trainingOption = scenario switch
        {
            ScenarioType.Classification => new ClassificationTrainingOptionV2
            {
                Subsampling = false,
                TrainingTime = trainingTime,
                LabelColumn = labelColumn,
                AvailableTrainers = trainers,
                ValidationOption = validationOption
            },
            ScenarioType.Regression => new RegressionTrainingOptionV2
            {
                Subsampling = false,
                TrainingTime = trainingTime,
                LabelColumn = labelColumn,
                AvailableTrainers = trainers,
                ValidationOption = validationOption
            },
            _ => null
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