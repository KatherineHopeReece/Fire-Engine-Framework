import geopandas as gpd

path = (
    "/Users/Martyn/GitRepos/Fire-Engine-Framework-dev/"
    "ignacio/LaughTests/Constant_Euler_Output/"
    "all_perimeters_history.gpkg"
)

gdf = gpd.read_file(path)

print("\nMissing values:")
print(gdf[["t_min", "area_ha"]].isna().sum())

print("\nRecords containing missing values:")
print(
    gdf.loc[
        gdf[["t_min", "area_ha"]].isna().any(axis=1),
        ["step", "t_min", "area_ha"]
    ].to_string(index=False)
)