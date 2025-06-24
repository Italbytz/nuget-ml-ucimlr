using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class
    CarEvaluationDataset : Dataset<CarEvaluationDataset.CarEvaluationModelInput>
{
    public override IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType,
        IEstimator<ITransformer> estimator, bool custom = false)
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
        return LoadFromTextFile<CarEvaluationModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    public class CarEvaluationModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"buying")]
        public string Buying { get; set; }

        [LoadColumn(1)] [ColumnName(@"maint")] public string Maint { get; set; }

        [LoadColumn(2)] [ColumnName(@"doors")] public string Doors { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"persons")]
        public string Persons { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"lug_boot")]
        public string Lug_boot { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"safety")]
        public string Safety { get; set; }

        [LoadColumn(6)] [ColumnName(@"class")] public string Class { get; set; }
    }
}