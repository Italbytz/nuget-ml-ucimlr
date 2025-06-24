using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class
    BalanceScaleDataset : Dataset<BalanceScaleDataset.BalanceScaleModelInput>
{
    public override IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType,
        IEstimator<ITransformer> estimator)
    {
        throw new NotImplementedException();
    }

    public override IDataView LoadFromTextFile(string path,
        char separatorChar = IDataset.TextLoaderDefaults.Separator,
        bool hasHeader = IDataset.TextLoaderDefaults.HasHeader,
        bool allowQuoting = IDataset.TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = IDataset.TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = IDataset.TextLoaderDefaults.AllowSparse)
    {
        return LoadFromTextFile<BalanceScaleModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
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