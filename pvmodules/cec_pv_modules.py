import pandas as pd
from PySAM.SixParsolve import new as SixParSolve
from PySAM import __version__
import datetime as dt
import typing as t
import logging
import numpy as np
from multiprocessing import Pool
from sys import argv


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


family_tech_pairs_to_remove = [
    ['Polycrystalline', 'Mono-c-Si'],
    ['Monocrystalline', 'Multi-c-Si'],
    ['CIGS Thin Film', 'Thin Film'],
    ['HIT-Si Thin Film', 'Multi-c-Si'],
    ['Thin Film', 'Multi-c-Si'],
    ['polycrystalline', 'Multi-c-Si'],
    ['Polycrystalline', 'Thin Film']
]


sam_cec_columns = ['Name', 'Manufacturer', 'Technology', 'Bifacial', 'STC', 'PTC', 'A_c',
       'Length', 'Width', 'N_s', 'I_sc_ref', 'V_oc_ref', 'I_mp_ref',
       'V_mp_ref', 'alpha_sc', 'beta_oc', 'T_NOCT', 'a_ref', 'I_L_ref',
       'I_o_ref', 'R_s', 'R_sh_ref', 'Adjust', 'gamma_r', 'BIPV', 'Version',
       'Date']


nan_row = [None] * 25


sixpar_model_to_sam_outputs_map = {
    'Adj': 'Adjust',
    'Il': 'I_L_ref',
    'Io': 'I_o_ref',
    'Rs': 'R_s',
    'Rsh': 'R_sh_ref',
    'a': 'a_ref'
}


sixpar_model_to_sam_cell_type_map = {
    'Mono-c-Si': 'monoSi',
    'Multi-c-Si': 'multiSi',
    'CdTe': 'cdte',
    'CIGS': 'cigs',
    'CIS': 'cis',
    'Thin Film': 'amorphous'
}


cec_to_sam_map = {
    'Nameplate Pmax': 'STC',
    'PTC': 'PTC',
    'A_c': 'A_c',
    'N_s': 'N_s',
    'BIPV': 'BIPV',
    'Nameplate Isc': 'I_sc_ref',
    'Nameplate Voc': 'V_oc_ref',
    'Nameplate Ipmax': 'I_mp_ref',
    'Nameplate Vpmax': 'V_mp_ref',
    'αIsc': 'alpha_sc',
    'βVoc': 'beta_oc',
    'γPmax': 'gamma_r',
    'Average NOCT': 'T_NOCT',
    'Long Side': 'Length',
    'Short Side': 'Width',
}


class SixParSolverExecutionError(ValueError):
    pass


def load_and_prepare_cec_dataset(file_path: str) -> pd.DataFrame:
    logger.info(f'preparing CEC dataset from {file_path}')
    cec_modules = pd.read_excel(file_path, engine='openpyxl', skiprows=[*range(16), 17], usecols=range(37))
    cec_modules = cec_modules.loc[~cec_modules['Manufacturer'].isna()]
    cec_modules['Description'] = cec_modules['Description'].replace(np.nan, '')
    for family, tech in family_tech_pairs_to_remove:
        cec_modules = cec_modules.loc[~((cec_modules['Family'] == family) & (cec_modules['Technology'] == tech))]
    logger.info(f'CEC dataset prepared')
    return cec_modules


def generate_module_coefficients(sam_row: dict) -> t.Optional[dict]:
    pv_solve = SixParSolve()
    pv_solve.SixParameterSolver.assign({
        'Imp': sam_row['I_mp_ref'],
        'Isc': sam_row['I_sc_ref'],
        'Nser': sam_row['N_s'],
        'Tref': 25,
        'Vmp': sam_row['V_mp_ref'],
        'Voc': sam_row['V_oc_ref'],
        'alpha_isc': sam_row['alpha_sc'] * 1e-2 * sam_row['I_mp_ref'],
        'beta_voc': sam_row['beta_oc'] * 1e-2 * sam_row['V_oc_ref'],
        'celltype': sixpar_model_to_sam_cell_type_map[sam_row['Technology']],
        'gamma_pmp': sam_row['gamma_r']
    })
    try:
        pv_solve.execute(0)
    except Exception as e:
        raise SixParSolverExecutionError(e)
    return pv_solve.Outputs.export()


def cec_to_sam(cec_row: pd.Series) -> dict:
    sam_row = {cec_to_sam_map[k]: cec_row[k] for k in cec_to_sam_map}
    sam_row['Name'] = f"{cec_row['Manufacturer']} {cec_row['Model Number']}"
    sam_row['Manufacturer'] = cec_row['Manufacturer']
    sam_row['Bifacial'] = int(
        'Bif' in cec_row['Description'] or 'bif' in cec_row['Description']
    )
    sam_row['Technology'] = 'CIS' if 'CIS' in cec_row['Technology'] else cec_row['Technology']
    return sam_row


def convert_row(cec_row: pd.Series) -> pd.Series:
    sam_row = cec_to_sam(cec_row)
    try:
        coefficients = generate_module_coefficients(sam_row)
    except SixParSolverExecutionError as e:
        # TODO: lock log file for multiprocessing?
        logger.warning(f'coefficient generation for {sam_row["Name"]} failed because {str(e).strip()}')
        sam_row.update({k: None for k in sixpar_model_to_sam_outputs_map.values()})
        return pd.Series(sam_row)
    sam_row.update({
        sixpar_model_to_sam_outputs_map[k]: coefficients[k] for k in sixpar_model_to_sam_outputs_map
    })
    return pd.Series(sam_row)


def get_writer(file_path: str) -> t.Callable:
    def writer(df: pd.DataFrame):
        try:
            df['Version'] = __version__
            df['Date'] = str(dt.date.today())
            df = df.loc[~df['Name'].isna(), sam_cec_columns]
            df.to_csv(file_path, index=False, mode='a', header=False)
        except Exception as e:
            logger.error(e)
            raise e
    return writer


def main(input_path: str, output_path: str, processes: int):
    cec_dataset = load_and_prepare_cec_dataset(input_path)
    cec_chunks = np.array_split(cec_dataset, cec_dataset.shape[0] // 10)
    logger.info(f'processing {cec_dataset.shape[0]} CEC modules in {len(cec_chunks)} chunks')
    pd.DataFrame(columns=sam_cec_columns).to_csv(output_path, header=True, index=False)
    writer = get_writer(output_path)

    with Pool(processes) as pool:
        jobs = []
        for chunk in cec_chunks:
            jobs.append(pool.apply_async(chunk.apply, (convert_row,), {'axis': 1}))
        i = 0
        for job in jobs:
            writer(job.get())
            i += 1
            if i % 100 == 0:
                logger.info(f'{i} modules completed')


if __name__ == '__main__':
    if len(argv) == 4:
        n_procs = int(argv[3])
    else:
        n_procs = 1
    main(argv[1], argv[2], n_procs)
