using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.UCIMLR;

public class
    WineQualityDataset : Dataset<WineQualityDataset.WineQualityModelInput>
{
    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Wine_Quality.csv";

    protected override string FilePrefix { get; } = "wine_quality";

    public override string? LabelColumnName { get; } = @"quality";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "fixed_acidity",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "volatile_acidity",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "citric_acid",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "residual_sugar",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "chlorides",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "free_sulfur_dioxide",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "total_sulfur_dioxide",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "density",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "pH",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "sulphates",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "alcohol",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "quality",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "Single",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          }
        ]
        """;

    public override IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType,
        IEstimator<ITransformer> estimator)
    {
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"fixed_acidity", @"fixed_acidity"),
                new InputOutputColumnPair(@"volatile_acidity",
                    @"volatile_acidity"),
                new InputOutputColumnPair(@"citric_acid", @"citric_acid"),
                new InputOutputColumnPair(@"residual_sugar", @"residual_sugar"),
                new InputOutputColumnPair(@"chlorides", @"chlorides"),
                new InputOutputColumnPair(@"free_sulfur_dioxide",
                    @"free_sulfur_dioxide"),
                new InputOutputColumnPair(@"total_sulfur_dioxide",
                    @"total_sulfur_dioxide"),
                new InputOutputColumnPair(@"density", @"density"),
                new InputOutputColumnPair(@"pH", @"pH"),
                new InputOutputColumnPair(@"sulphates", @"sulphates"),
                new InputOutputColumnPair(@"alcohol", @"alcohol")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features",
                @"fixed_acidity", @"volatile_acidity", @"citric_acid",
                @"residual_sugar", @"chlorides", @"free_sulfur_dioxide",
                @"total_sulfur_dioxide", @"density", @"pH", @"sulphates",
                @"alcohol"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"quality",
                @"quality", addKeyValueAnnotationsAsText: false))
            .Append(estimator)
            .Append(
                mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel",
                    @"PredictedLabel"));

        return pipeline;
    }

    public override IDataView LoadFromTextFile(string path,
        char separatorChar = IDataset.TextLoaderDefaults.Separator,
        bool hasHeader = IDataset.TextLoaderDefaults.HasHeader,
        bool allowQuoting = IDataset.TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = IDataset.TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = IDataset.TextLoaderDefaults.AllowSparse)
    {
        return LoadFromTextFile<WineQualityModelInput>(path, separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    public class WineQualityModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"fixed_acidity")]
        public float Fixed_acidity { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"volatile_acidity")]
        public float Volatile_acidity { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"citric_acid")]
        public float Citric_acid { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"residual_sugar")]
        public float Residual_sugar { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"chlorides")]
        public float Chlorides { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"free_sulfur_dioxide")]
        public float Free_sulfur_dioxide { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"total_sulfur_dioxide")]
        public float Total_sulfur_dioxide { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"density")]
        public float Density { get; set; }

        [LoadColumn(8)] [ColumnName(@"pH")] public float PH { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"sulphates")]
        public float Sulphates { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"alcohol")]
        public float Alcohol { get; set; }

        [LoadColumn(11)]
        [ColumnName(@"quality")]
        public float Quality { get; set; }
    }
}