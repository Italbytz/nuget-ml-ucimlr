using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class
    BanknoteAuthenticationDataset : Dataset<
    BanknoteAuthenticationDataset.BanknoteAuthenticationInput>
{
    private readonly LookupMap<float>[] _lookupData =
    [
        new(0.0f),
        new(1.0f)
    ];

    public override bool HasHeader { get; } = true;

    public override char Separator { get; } = ',';

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Banknote_Authentication.csv";

    public override string FilePrefix { get; } = "banknote";

    public override string? LabelColumnName { get; } = @"class";


    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "variance",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "skewness",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "curtosis",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "entropy",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "class",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "Single",
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
        return LoadFromTextFile<BanknoteAuthenticationInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer>? BuildLabelMappingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return processingType switch
        {
            ProcessingType.FeatureBinningAndCustomLabelMapping => mlContext
                .Transforms.Conversion.MapValueToKey(@"Label", @"class",
                    keyData: mlContext.Data.LoadFromEnumerable(_lookupData)),
            ProcessingType.Standard => mlContext.Transforms.Conversion
                .MapValueToKey(@"class",
                    @"class", addKeyValueAnnotationsAsText: false)
                .Append(mlContext.Transforms.CopyColumns("Label", "class")),
            _ => throw new NotImplementedException()
        };
    }

    protected override IEstimator<ITransformer>? BuildFeaturizationPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        if (processingType == ProcessingType.Standard)
            return mlContext.Transforms.Concatenate(@"Features",
                @"variance", @"skewness", @"curtosis", @"entropy");

        if (processingType ==
            ProcessingType.FeatureBinningAndCustomLabelMapping)
            return mlContext.Transforms.NormalizeBinning(new[]
                {
                    new InputOutputColumnPair(@"variance", @"variance"),
                    new InputOutputColumnPair(@"skewness", @"skewness"),
                    new InputOutputColumnPair(@"curtosis", @"curtosis"),
                    new InputOutputColumnPair(@"entropy", @"entropy")
                }, maximumBinCount: 4)
                .Append(mlContext.Transforms.Concatenate(@"Features",
                    @"variance", @"skewness", @"curtosis", @"entropy"));
        throw new NotImplementedException();
    }

    protected override IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return
            mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"variance", @"variance"),
                new InputOutputColumnPair(@"skewness", @"skewness"),
                new InputOutputColumnPair(@"curtosis", @"curtosis"),
                new InputOutputColumnPair(@"entropy", @"entropy")
            });
    }

    public class BanknoteAuthenticationInput
    {
        [LoadColumn(0)]
        [ColumnName(@"variance")]
        public float Variance { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"skewness")]
        public float Skewness { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"curtosis")]
        public float Curtosis { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"entropy")]
        public float Entropy { get; set; }

        [LoadColumn(4)] [ColumnName(@"class")] public float Class { get; set; }
    }
}