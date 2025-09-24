using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class IrisDataset : Dataset<IrisDataset.IrisModelInput>
{
    private readonly LookupMap<string>[] _lookupData =
    [
        new("Iris-setosa"),
        new("Iris-versicolor"),
        new("Iris-virginica")
    ];

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

    public override IDataView LoadFromTextFile(string path,
        char? separatorChar = null,
        bool? hasHeader = null, bool? allowQuoting = null,
        bool? trimWhitespace = null, bool? allowSparse = null)
    {
        return LoadFromTextFile<IrisModelInput>(path, separatorChar, hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer>? BuildLabelMappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (processingType == ProcessingType.Standard)
            return mlContext.Transforms.Conversion.MapValueToKey(
                    @"class",
                    @"class", addKeyValueAnnotationsAsText: false)
                .Append(mlContext.Transforms.CopyColumns("Label", "class"));

        if (processingType ==
            ProcessingType.FeatureBinningAndCustomLabelMapping)
            return mlContext.Transforms.Conversion.MapValueToKey(
                @"Label",
                @"class",
                keyData: mlContext.Data.LoadFromEnumerable(_lookupData));

        throw new NotImplementedException();
    }

    protected override IEstimator<ITransformer>? BuildFeaturizationPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (processingType == ProcessingType.Standard)
            return mlContext.Transforms.Concatenate(@"Features",
                @"sepal length", @"sepal width", @"petal length",
                @"petal width");

        if (processingType ==
            ProcessingType.FeatureBinningAndCustomLabelMapping)
            return mlContext.Transforms.NormalizeBinning(new[]
                {
                    new InputOutputColumnPair(@"sepal length",
                        @"sepal length"),
                    new InputOutputColumnPair(@"sepal width",
                        @"sepal width"),
                    new InputOutputColumnPair(@"petal length",
                        @"petal length"),
                    new InputOutputColumnPair(@"petal width",
                        @"petal width")
                }, maximumBinCount: 4)
                .Append(mlContext.Transforms.Concatenate(@"Features",
                    @"sepal length", @"sepal width", @"petal length",
                    @"petal width"));
        throw new NotImplementedException();
    }

    protected override IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (scenarioType == ScenarioType.Classification)
            return mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"sepal length",
                    @"sepal length"),
                new InputOutputColumnPair(@"sepal width",
                    @"sepal width"),
                new InputOutputColumnPair(@"petal length",
                    @"petal length"),
                new InputOutputColumnPair(@"petal width",
                    @"petal width")
            });

        throw new NotSupportedException(
            $"The scenario type {scenarioType} is not supported.");
    }

    protected override IEstimator<ITransformer>? BuildLabelRemappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (scenarioType == ScenarioType.Classification)
        {
            if (processingType ==
                ProcessingType.FeatureBinningAndCustomLabelMapping) return null;
            return mlContext.Transforms.Conversion.MapKeyToValue(
                @"PredictedLabel", @"PredictedLabel");
        }

        throw new NotSupportedException(
            $"The scenario type {scenarioType} is not supported.");
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