using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class
    BalanceScaleDataset : Dataset<BalanceScaleDataset.BalanceScaleModelInput>
{
    private readonly LookupMap<string>[] _lookupData =
    [
        new("B"),
        new("R"),
        new("L")
    ];

    public override bool HasHeader { get; } = true;

    public override char Separator { get; } = ',';

    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.balancescale.csv";

    public override string FilePrefix { get; } = "balancescale";

    public override string? LabelColumnName { get; } = @"class";

    public override IDataView LoadFromTextFile(string path,
        char? separatorChar = null,
        bool? hasHeader = null, bool? allowQuoting = null,
        bool? trimWhitespace = null, bool? allowSparse = null)
    {
        return LoadFromTextFile<BalanceScaleModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return mlContext.Transforms.ReplaceMissingValues(new[]
        {
            new InputOutputColumnPair(@"right-distance",
                @"right-distance"),
            new InputOutputColumnPair(@"right-weight", @"right-weight"),
            new InputOutputColumnPair(@"left-distance",
                @"left-distance"),
            new InputOutputColumnPair(@"left-weight", @"left-weight")
        });
    }

    protected override IEstimator<ITransformer>? BuildFeaturizationPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        return mlContext.Transforms.Concatenate(@"Features", @"right-distance",
            @"right-weight", @"left-distance", @"left-weight");
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
                .MapValueToKey(@"class", @"class",
                    addKeyValueAnnotationsAsText: false)
                .Append(mlContext.Transforms.CopyColumns("Label", "class")),
            _ => throw new NotImplementedException()
        };
    }


    public class BalanceScaleModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"right-distance")]
        public float Right_distance { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"right-weight")]
        public float Right_weight { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"left-distance")]
        public float Left_distance { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"left-weight")]
        public float Left_weight { get; set; }

        [LoadColumn(4)] [ColumnName(@"class")] public string Class { get; set; }
    }
}