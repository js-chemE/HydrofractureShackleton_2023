import os
from pydantic import BaseModel, Field
import rioxarray as rioxr
import rasterio
import numpy as np

from typing import Any

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AngleResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    name: str | None = Field(default = None, description="Name of the tile")
    file_vx: str | None = Field(default = None, description="Path to the vx file")
    file_vy: str | None = Field(default = None, description="Path to the vy file")
    file_alpha_c: str | None = Field(default = None, description="Path to the alpha_c file")
    vx: Any | None = Field(default = None, description="vx data")
    vy: Any | None = Field(default = None, description="vy data")
    alpha_v: Any | None = Field(default = None, description="velocity of angl")
    dx: float | None = Field(default = None, description="delta x")
    dy: float | None = Field(default = None, description="delta y")

    alpha_c: Any | None = Field(default = None, description="alpha_c data")
    theta_p: Any | None = Field(default = None, description="theta_p data")
    emax: Any | None = Field(default = None, description="emax data")
    emin: Any | None = Field(default = None, description="emin data")
    alpha_p: Any | None = Field(default = None, description="alpha_p data")
    theta_p_degr: Any | None = Field(default = None, description="alpha_p in degrees")

    alpha_v_match: Any | None = Field(default = None, description="alpha_v data")
    theta_p_match: Any | None = Field(default = None, description="theta_p data")

    delta_theta: Any | None = Field(default = None, description="delta theta data")
    delta_alpha: Any | None = Field(default = None, description="delta alpha data")
    def calculate(self) -> None:
        """
        Calculate the delta velocity for the fracture.
        """
        self.vx =rioxr.open_rasterio(self.file_vx)
        self.vy = rioxr.open_rasterio(self.file_vy)
        self.alpha_v = np.arctan(self.vy / self.vx) / (2 * np.pi) * 360
        self.dx = np.unique(np.diff(self.vx["x"].values))[0]
        self.dy = np.unique(np.diff(self.vy["y"].values))[0]

        self.alpha_c = rioxr.open_rasterio(self.file_alpha_c)

        # Prinicipal strain values 

        pix = 1  # number of pixels to shift
        res = self.dx # spatial resolution of grid

        dudx = (self.vx - self.vx.roll(x=pix) )/res
        dvdy = (self.vy - self.vy.roll(y=pix) )/res
        dudy = (self.vx - self.vx.roll(y=pix) )/res
        dvdx = (self.vy - self.vy.roll(x=pix) )/res
        exx = 0.5*(dudx+dudx)
        eyy = 0.5*(dvdy+dvdy)
        exy = 0.5*(dudy+dvdx)
        emax_xr = (exx+eyy)*0.5 + np.sqrt(np.power(exx-eyy,2)*0.25+np.power(exy,2))
        self.emax = emax_xr.squeeze()

        emin_xr = (exx+eyy)*0.5 - np.sqrt(np.power(exx-eyy,2)*0.25+np.power(exy,2))
        self.emin = emin_xr.squeeze()

        # Orientation of principal strain
        # self.emax.rio.to_raster(raster_path= os.path.join(settings.region_folder, self.name + "_emax.tif"))
        # self.emin.rio.to_raster(raster_path= os.path.join(settings.region_folder, self.name + "_emin.tif"))

        self.theta_p = (np.arctan(2*exy/(exx-eyy))/2) # in radiations
        #  Check if theta_p aligns with e_min or e_max, and update theta_p value + 0.25 pi
        idx_e22_eq_emax = check_theta_p_e11_e22(exx,eyy,exy,self.theta_p, self.emax)
        self.theta_p = self.theta_p.squeeze().values
        self.theta_p[idx_e22_eq_emax] +=  np.pi/4 # 90 degrees is 1/4 pi 

        theta_p_degr = self.theta_p*360/(2*np.pi)
        # convert to xarray dataArray
        self.theta_p_degr = emax_xr.copy(data=np.expand_dims(theta_p_degr,axis=0))

        self.theta_p_match = self.theta_p_degr.rio.reproject_match(self.alpha_c, resampling=rasterio.enums.Resampling.nearest,nodata=np.nan) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308
        self.alpha_v_match = self.alpha_v.rio.reproject_match(self.alpha_c,resampling=rasterio.enums.Resampling.nearest,nodata=np.nan) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308

        self.delta_theta = self.theta_p_match - self.alpha_c
        self.delta_alpha = self.alpha_v_match - self.alpha_c
    
    def export(self, basename: str, path: str) -> None:
        """
        Export the results to a file.
        """
        logger.info(f"Exporting >delta-alpha<, >emax<, >emin<, >delta-theta<, for >{basename}< to {path}")
        self.delta_alpha.rio.to_raster(raster_path=os.path.join(path, basename + "_delta-alpha.tif"))
        self.emax.rio.to_raster(raster_path=os.path.join(path, basename + "_emax.tif"))
        self.emin.rio.to_raster(raster_path=os.path.join(path, basename + "_emin.tif"))
        self.delta_theta.rio.to_raster(raster_path=os.path.join(path, basename + "_delta-theta.tif"))

def calculate_delta_velocity_fracture(
        file_vx: str,
        file_vy: str,
        file_alpha_c: str
    ) -> AngleResult:
    
    result = AngleResult(
        file_vx=file_vx,
        file_vy=file_vy,
        file_alpha_c=file_alpha_c,
    )
    result.calculate()
    return result

# Check if theta_p aligns with e_max or e_min
def check_theta_p_e11_e22(exx,eyy,exy,theta_p, emax):
        
    E = np.array([ [exx.squeeze(), exy.squeeze() ], 
                [exy.squeeze(), eyy.squeeze() ] ]) # shape (x y n k) (2, 2, 494, 401) <-- should be (494, 401, 2, 2)

    Q = np.array([ [ np.cos(theta_p.squeeze()), np.sin(theta_p.squeeze())], 
                [-np.sin(theta_p.squeeze()), np.cos(theta_p.squeeze())] ])  # shape (x y k m) (2, 2, 494, 401) <-- should be (494, 401, 2, 2)      

    Qt = np.transpose(Q, (1, 0, 2, 3)) # (2, 2, 494, 401); transposed the (2,2) axes

    # move axes for np.matmul (from (2,2,x,y) to (x,y, 2, 2 ))
    E = np.moveaxis(E, [0, 1], [-2, -1]) # shape (494, 401, 2, 2); (x y n k)
    Q = np.moveaxis(Q, [0, 1], [-2, -1]) # shape (494, 401, 2, 2); (x y k m)
    Qt = np.moveaxis(Qt, [0, 1], [-2, -1]) #  shape (494, 401, 2, 2); (x y k m) 

    Edot = np.matmul(np.matmul(Q,E),Qt)

    # check single element:
    # i=2
    # j=6
    # print('Example: ')
    # print('emax {}, emin {}' .format(emax.squeeze()[i,j].values, emin.squeeze()[i,j].values))
    # print('E11: {}, E22: {}'.format(Edot[i,j,0,0],Edot[i,j,1,1]))

    # Update_theta_p to match e11
    e11 = Edot[:,:,0,0]
    e22 = Edot[:,:,1,1] 
    ndec = 6

    # idx where emax=e11, theta_p is correct 
    idx_e11_eq_emax = e11.round(ndec) == emax.values.round(ndec) 

    # idx where emax=e22, theta_p should be + 90 degree
    idx_e22_eq_emax = e22.round(ndec) == emax.values.round(ndec) 

    # if emax is not e11, it should be e22
    idx_correct = ~idx_e11_eq_emax == idx_e22_eq_emax 
    idx_correct[np.isnan(e11)] = True
    # --> check pixels where this is not true
    if not idx_correct.all():
        diff_e11 = np.nan_to_num(np.abs(emax-e11))
        diff_e22 = np.nan_to_num(np.abs(emax-e22)) 
        min_diff =  np.array((diff_e11,diff_e22)).min(axis=0)
        mindiff_e11_or_e22 = np.array((diff_e11,diff_e22)).argmin(axis=0) # 0 of minimum is found in e11; 1 if minimum diff is found in e22
        mindiff_e11_or_e22[idx_correct] = -999                # skip pixels that were already correct

        idx_e11_eq_emax[np.where(mindiff_e11_or_e22 == 0)] = True
        idx_e22_eq_emax[np.where(mindiff_e11_or_e22 == 1)] = True

        idx_correct = ~idx_e11_eq_emax == idx_e22_eq_emax 
        idx_correct[np.isnan(e11)] = True

    return idx_e22_eq_emax
