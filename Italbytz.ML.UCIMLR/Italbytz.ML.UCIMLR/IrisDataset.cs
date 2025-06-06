using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class IrisDataset : Dataset<IrisDataset.IrisModelInput>
{
    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Iris.csv";

    public override string FilePrefix { get; } = "iris";

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

    public override IDataView LoadFromTextFile(string path,
        char separatorChar = IDataset.TextLoaderDefaults.Separator,
        bool hasHeader = IDataset.TextLoaderDefaults.HasHeader,
        bool allowQuoting = IDataset.TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = IDataset.TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = IDataset.TextLoaderDefaults.AllowSparse)
    {
        return LoadFromTextFile<IrisModelInput>(path, separatorChar, hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    /// <summary>
    ///     Represents the input data schema for the Iris dataset used in ML.NET
    ///     models.
    /// </summary>
    public class IrisModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"sepal length")]
        public float Sepal_length { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"sepal width")]
        public float Sepal_width { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"petal length")]
        public float Petal_length { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"petal width")]
        public float Petal_width { get; set; }

        [LoadColumn(4)] [ColumnName(@"class")] public string Class { get; set; }
    }
}