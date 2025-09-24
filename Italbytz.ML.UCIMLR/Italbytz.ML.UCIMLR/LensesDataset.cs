using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class LensesDataset : Dataset<LensesDataset.LensesModelInput>
{
    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.lenses.csv";

    public override string FilePrefix { get; } = "lenses";

    public override string? LabelColumnName { get; } = @"class";

    public override IDataView LoadFromTextFile(string path,
        char? separatorChar = null,
        bool? hasHeader = null, bool? allowQuoting = null,
        bool? trimWhitespace = null, bool? allowSparse = null)
    {
        return LoadFromTextFile<LensesModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    protected override IEstimator<ITransformer> AdditionalPreprocessingPipeline(
        MLContext mlContext,
        ScenarioType scenarioType = ScenarioType.Classification,
        ProcessingType processingType = ProcessingType.Standard)
    {
        throw new NotImplementedException();
    }

    public class LensesModelInput
    {
        [LoadColumn(0)] [ColumnName(@"age")] public float Age { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"spectacle_prescription")]
        public float Spectacle_prescription { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"astigmatic")]
        public float Astigmatic { get; set; }

        [LoadColumn(3)] [ColumnName(@"class")] public uint Class { get; set; }
    }
}