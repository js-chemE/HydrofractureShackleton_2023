import shapely.ops
import shapely.wkt
from shapely.geometry import LineString, MultiLineString, Point, Polygon, mapping, shape

import app.sgl_processor as sgl
import app.sgl_utils

if __name__ == "__main__":
    settings = sgl.ProcessorSGLSettings()
    settings.parent_folder = r""
    settings.rangestart = "20181201"
    settings.rangeend = "20190331"

    sgl.initialize(settings)
    mode = 1

    if mode == 0:
        """Testing Field"""
        metas = sgl.get_metadata_from_csv(
            settings.season_folder, ["tile-183", "tile-182", "tile-181"]
        )
        coll = sgl.create_collection_from_meta(settings, metas)
        sgl.post_process_tifs(
            settings,
            coll,
            max_windows=100,
            print_metadata=True,
            update_csv=True,
            update_tif=True,
        )
        sgl.export_max_lake_extents(settings.season_folder, coll, print_metadata=True)
        sgl.combine_max_lake_extents(settings.season_folder, coll, print_metadata=True)
        sgl.add_extent_statistics(
            settings.season_folder, coll, update_bool=True, print_bool=True
        )
        sgl.add_iceshelf_statistics(
            settings.season_folder, coll, update_bool=True, print_bool=True
        )
        sgl.export_drainages(settings, coll, print_metadata=True)
        sgl.combine_filter_drainages(settings, coll, print_metadata=True)
        pass

    elif mode == 1:
        """Read WindowCollection csv and post process raw tifs"""
        (
            files_tif,
            files_csv,
            files_shp,
            tile_names,
        ) = app.sgl_utils.read_files_from_folder(settings.season_folder)
        sgl.process_and_export_csv(
            settings, files_csv, names=["tile-183", "tile-182", "tile-181"]
        )

    elif mode == 2:
        """Read WindowCollection csv and post process raw tifs"""
        metas = sgl.get_metadata_from_csv(
            settings.season_folder, ["tile-183", "tile-182", "tile-181"]
        )
        coll = sgl.create_collection_from_meta(settings, metas)
        sgl.post_process_tifs(
            settings,
            coll,
            max_windows=100,
            print_metadata=True,
            update_csv=True,
            update_tif=True,
        )

    elif mode == 3:
        metas = sgl.get_metadata_from_csv(
            settings.season_folder, ["tile-183", "tile-182", "tile-181"]
        )
        coll = sgl.create_collection_from_meta(settings, metas)
        # sgl.export_max_lake_extents(settings.season_folder, coll, print_metadata = True)
        # sgl.combine_max_lake_extents(settings.season_folder, coll, print_metadata = True)
        sgl.add_extent_statistics(
            settings.season_folder, coll, update_bool=True, print_bool=True
        )
        sgl.add_iceshelf_statistics(
            settings.season_folder, coll, update_bool=True, print_bool=True
        )

    elif mode == 4:
        metas = sgl.get_metadata_from_csv(
            settings.season_folder, ["tile-183", "tile-182", "tile-181"]
        )
        coll = sgl.create_collection_from_meta(settings, metas)
        (
            files_tif,
            files_csv,
            files_shp,
            tile_names,
        ) = app.sgl_utils.read_files_from_folder(settings.region_folder)
        # sgl.create_roi_csv(settings.region.lower(), settings.region_folder, files_shp)
        roi_coll = sgl.create_roicollection_from_csv(settings, "shackleton1920")
        sgl.add_roi_statistics(settings, roi_coll, coll, update_bool=True)

    elif mode == 5:
        metas = sgl.get_metadata_from_csv(
            settings.season_folder, ["tile-183", "tile-182", "tile-181"]
        )
        coll = sgl.create_collection_from_meta(settings, metas)
        # dmgs = sgl.get_vectorized_dmgs(settings.season_folder, coll, print_bool = True, category_bins = 5)
        lakeextents = sgl.get_vectorized_lakeextents(
            settings.season_folder, coll, print_bool=True
        )

    elif mode == 6:
        """Exporting Drainages"""
        metas = sgl.get_metadata_from_csv(
            settings.season_folder, ["tile-183", "tile-182", "tile-181"]
        )
        coll = sgl.create_collection_from_meta(settings, metas)
        sgl.export_drainages(settings, coll, print_metadata=True)

    elif mode == 7:
        """Exporting Drainages"""
        metas = sgl.get_metadata_from_csv(
            settings.season_folder, ["tile-183", "tile-182", "tile-181"]
        )
        coll = sgl.create_collection_from_meta(settings, metas)
        sgl.combine_filter_drainages(
            settings, coll, print_metadata=True, min_median_depth=0.5, min_std=0.20
        )

    else:
        print("No matching mode selected")
