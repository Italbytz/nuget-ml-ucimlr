using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;

namespace Italbytz.ML.UCIMLR;

public class IrisDataset : Dataset
{
    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Iris.csv";

    protected override string FilePrefix { get; } = "iris";

    public override string? LabelColumnName { get; } = @"class";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "sepal length",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "sepal width",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "petal length",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "petal width",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "class",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          }
        ]
        """;

    public override IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType, IEstimator<ITransformer> trainer)
    {
        if (scenarioType == ScenarioType.Classification)
        {
            var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
                {
                    new InputOutputColumnPair(@"sepal length", @"sepal length"),
                    new InputOutputColumnPair(@"sepal width", @"sepal width"),
                    new InputOutputColumnPair(@"petal length", @"petal length"),
                    new InputOutputColumnPair(@"petal width", @"petal width")
                })
                .Append(mlContext.Transforms.Concatenate(@"Features",
                    @"sepal length", @"sepal width", @"petal length",
                    @"petal width"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(@"class",
                    @"class", addKeyValueAnnotationsAsText: false))
                .Append(
                    trainer)
                .Append(
                    mlContext.Transforms.Conversion.MapKeyToValue(
                        @"PredictedLabel", @"PredictedLabel"));

            return pipeline;
        }

        throw new NotSupportedException(
            $"The scenario type {scenarioType} is not supported.");
    }

    protected override IDataView? LoadFromTextFile(string tempFile)
    {
        var mlContext = new MLContext();

        // Load the dataset from the temporary file
        var data = mlContext.Data.LoadFromTextFile<IrisModelInput>(
            tempFile, ',', true);

        return data;
    }
}