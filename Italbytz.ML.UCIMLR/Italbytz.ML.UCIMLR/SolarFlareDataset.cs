using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class SolarFlareDataset : Dataset<SolarFlareDataset.SolarflareModelInput>
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
        return LoadFromTextFile<SolarflareModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    public class SolarflareModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"modified Zurich class")]
        public string Modified_Zurich_class { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"largest spot size")]
        public string Largest_spot_size { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"spot distribution")]
        public string Spot_distribution { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"activity")]
        public float Activity { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"evolution")]
        public float Evolution { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"previous 24 hour flare activity")]
        public float Previous_24_hour_flare_activity { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"historically-complex")]
        public float Historically_complex { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"became complex on this pass")]
        public float Became_complex_on_this_pass { get; set; }

        [LoadColumn(8)] [ColumnName(@"area")] public float Area { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"area of largest spot")]
        public float Area_of_largest_spot { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"flares")]
        public uint Flares { get; set; }
    }
}