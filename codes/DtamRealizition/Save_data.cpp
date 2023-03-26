//  NB this is a file of notes, with code fragments from LSD-SLAM and Morphogenesis




// from openCV Viz module
void cv::viz::writeCloud(const String& file, InputArray cloud, InputArray colors, InputArray normals, bool binary)
{
    CV_Assert(file.size() > 4 && "Extension is required");
    String extension = file.substr(file.size()-4);

    vtkSmartPointer<vtkCloudMatSource> source = vtkSmartPointer<vtkCloudMatSource>::New();
    source->SetColorCloudNormals(cloud, colors, normals);

    vtkSmartPointer<vtkWriter> writer;
    if (extension == ".xyz")
    {
        writer = vtkSmartPointer<vtkXYZWriter>::New();
        vtkXYZWriter::SafeDownCast(writer)->SetFileName(file.c_str());
    }
    else if (extension == ".ply")
    {
        writer = vtkSmartPointer<vtkPLYWriter>::New();
        vtkPLYWriter::SafeDownCast(writer)->SetFileName(file.c_str());
        vtkPLYWriter::SafeDownCast(writer)->SetFileType(binary ? VTK_BINARY : VTK_ASCII);
        vtkPLYWriter::SafeDownCast(writer)->SetArrayName("Colors");
    }
    else if (extension == ".obj")
    {
        writer = vtkSmartPointer<vtkOBJWriter>::New();
        vtkOBJWriter::SafeDownCast(writer)->SetFileName(file.c_str());
    }
    else
        CV_Error(Error::StsError, "Unsupported format");

    writer->SetInputConnection(source->GetOutputPort());
    writer->Write();
}



// from Morphogenesis
void FluidSystem::SavePointsCSV2 ( const char * relativePath, int frame ){
    if (m_FParams.debug>1) std::cout << "\n  SavePointsCSV2 ( const char * relativePath = "<< relativePath << ", int frame = "<< frame << " );  started \n" << std::flush;
    char buf[256];
    frame += 100000;    // ensures numerical and alphabetic order match
    sprintf ( buf, "%s/particles_pos_vel_color%04d.csv", relativePath, frame );
    FILE* fp = fopen ( buf, "w" );
    if (fp == NULL) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::SavePointsCSV ( const char * relativePath, int frame )  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }
    int numpnt = mMaxPoints;//NumPoints();
    Vector3DF* Pos;
    Vector3DF* Vel;
    float *Conc;
    uint* Age, *Clr, *NerveIdx, *ElastIdx, *Particle_Idx, *Particle_ID, *Mass_Radius, *EpiGen;                  // Q: why are these pointers? A: they get dereferenced below.
    uint mass, radius;
    float *ElastIdxPtr;

    fprintf(fp, "i,, x coord, y coord, z coord\t\t x vel, y vel, z vel\t\t age,  color\t\t FELASTIDX[%u*%u]", BONDS_PER_PARTICLE, DATA_PER_BOND);  // This system inserts commas to align header with csv data
    for (int i=0; i<BONDS_PER_PARTICLE; i++)fprintf(fp, ",(%u)[0]curIdx, [1]elastLim, [2]restLn, [3]modulus, [4]damping, [5]partID, [6]bond index, [7]stress integrator, [8]change-type,,  ",i);
    fprintf(fp, "\t");
    fprintf(fp, "\tParticle_ID, mass, radius, FNERVEIDX,\t\t Particle_Idx[%u*2]", BONDS_PER_PARTICLE);
    for (int i=0; i<BONDS_PER_PARTICLE; i++)fprintf(fp, "%u,,, ",i);
    fprintf(fp, "\t\tFCONC[%u] ", NUM_TF);
    for (int i=0; i<NUM_TF; i++)fprintf(fp, "%u, ",i);
    fprintf(fp, "\t\tFEPIGEN[%u] ", NUM_GENES);
    for (int i=0; i<NUM_GENES; i++)fprintf(fp, "%u, ",i);
    fprintf(fp, "\n");

    for(int i=0; i<numpnt; i++) {       // nb need get..() accessors for private data.
        Pos = getPos(i);                // e.g.  Vector3DF* getPos ( int n )	{ return &m_Fluid.bufV3(FPOS)[n]; }
        Vel = getVel(i);
        Age = getAge(i);
        Clr = getClr(i);
        ElastIdx = getElastIdx(i);      // NB [BONDS_PER_PARTICLE]
      //if (m_FParams.debug>1)printf("\t%u,",ElastIdx[0]);
        ElastIdxPtr = (float*)ElastIdx; // #############packing floats and uints into the same array - should replace with a struct.#################
        Particle_Idx = getParticle_Idx(i);
        Particle_ID = getParticle_ID(i);//# uint  original pnum, used for bonds between particles. 32bit, track upto 4Bn particles.
        if(*Particle_ID==0){
         if (m_FParams.debug>1) std::cout << "Particle_ID = pointer not assigned. i="<<i<<". \t" << std::flush;
         return;
        }
        // ? should I be splitting mass_radius with bitshift etc  OR just use two uit arrays .... where are/will these used anyway ?
        Mass_Radius = getMass_Radius(i);//# uint holding modulus 16bit and limit 16bit.
        if(*Mass_Radius==0){   mass = 0; }else{  mass = *Mass_Radius; }    // modulus          // '&' bitwise AND is bit masking. ;
        radius = mass >> 16;
        mass = mass & TWO_POW_16_MINUS_1;

        NerveIdx = getNerveIdx(i);      //# uint
        //Conc = getConc(i);              //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
        //EpiGen = getEpiGen(i);          //# uint[NUM_GENES]  see below.

        fprintf(fp, "%u,,%f,%f,%f,\t%f,%f,%f,\t %u, %u,, \t", i, Pos->x, Pos->y,Pos->z, Vel->x,Vel->y,Vel->z, *Age, *Clr );
        //if (m_FParams.debug>1) std::cout<<"\t"<<Pos->z<<std::flush;
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) {
            fprintf(fp, "%u, %f, %f, %f, %f, %u, %u, %f, %u, ", ElastIdx[j], ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4], ElastIdx[j+5], ElastIdx[j+6], ElastIdxPtr[j+7], ElastIdx[j+8] );

           /*
            // if ((j%DATA_PER_BOND==0)||((j+1)%DATA_PER_BOND==0))  fprintf(fp, "%u, ",  ElastIdx[j] );  // print as int   [0]current index, [5]particle ID, [6]bond index
           // else  fprintf(fp, "%f, ",  ElastIdxPtr[j] );                                              // print as float [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff,
           //  if((j+1)%DATA_PER_BOND==0)
            */
            fprintf(fp, "\t\t");
        }
        fprintf(fp, " \t%u, %u, %u, %u, \t\t", *Particle_ID, mass, radius, *NerveIdx );
        for(int j=0; j<(BONDS_PER_PARTICLE*2); j+=2)   { fprintf(fp, "%u, %u,, ",  Particle_Idx[j], Particle_Idx[j+1] );}  fprintf(fp, "\t\t"); // NB index of other particle AND other particle's index of the bond

        for(int j=0; j<(NUM_TF); j++)               {
            Conc = getConc(j);
            fprintf(fp, "%f, ",  Conc[i] );
        }fprintf(fp, "\t\t");

        for(int j=0; j<(NUM_GENES); j++)            {
            EpiGen = getEpiGen(j);
            fprintf(fp, "%u, ",  EpiGen[i] );   // NB FEPIGEN[gene][particle], for memory efficiency on the device. ? Need to test.
        }fprintf(fp, " \n");
    }
    fclose ( fp );
    fflush ( fp );
}






// from Morphogenesis
void FluidSystem::WriteSimParams ( const char * relativePath ){
    Vector3DF point_grav_pos, pplane_grav_dir, pemit_pos, pemit_rate, pemit_ang, pemit_dang, pvolmin, pvolmax, pinitmin, pinitmax;

    int pwrapx, pwall_barrier, plevy_barrier, pdrain_barrier, prun;

    point_grav_pos = m_Vec [ PPOINT_GRAV_POS ];
    pplane_grav_dir = m_Vec [ PPLANE_GRAV_DIR ];
    pemit_pos = m_Vec [ PEMIT_POS ];
    pemit_rate = m_Vec [ PEMIT_RATE ];
    pemit_ang = m_Vec [ PEMIT_ANG ];
    pemit_dang = m_Vec [ PEMIT_DANG ];
    pvolmin = m_Vec [ PVOLMIN ];
    pvolmax = m_Vec [ PVOLMAX ];
    pinitmin = m_Vec [ PINITMIN ];
    pinitmax = m_Vec [ PINITMAX ];

    pwrapx = m_Toggle [ PWRAP_X ] ;
    pwall_barrier =  m_Toggle [ PWALL_BARRIER ];
    plevy_barrier = m_Toggle [ PLEVY_BARRIER ];
    pdrain_barrier = m_Toggle [ PDRAIN_BARRIER ];
    prun = m_Toggle [ PRUN ];

    // open file to write SimParams to
    char SimParams_file_path[256];
    sprintf ( SimParams_file_path, "%s/SimParams.txt", relativePath );
    if (m_FParams.debug>1)printf("\n## opening file %s ", SimParams_file_path);
    FILE* SimParams_file = fopen ( SimParams_file_path, "w" );
    if (SimParams_file == NULL) {
        if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::WriteSimParams (const char * relativePath )  Could not open file "<< SimParams_file_path <<"\n"<< std::flush;
        assert(0);
    }

    int ret = std::fprintf(SimParams_file,
                           " m_Time = %f\n m_DT = %f\n m_Param [ PSIMSCALE ] = %f\n m_Param [ PGRID_DENSITY ] = %f\n m_Param [ PVISC ] = %f\n m_Param [ PRESTDENSITY ] = %f\n m_Param [ PSPACING ] = %f\n m_Param [ PMASS ] = %f\n m_Param [ PRADIUS ] = %f\n m_Param [ PDIST ] = %f\n m_Param [ PSMOOTHRADIUS ] = %f\n m_Param [ PINTSTIFF ] = %f\n m_Param [ PEXTSTIFF ] = %f\n m_Param [ PEXTDAMP ] = %f\n m_Param [ PACCEL_LIMIT ] = %f\n m_Param [ PVEL_LIMIT ] = %f\n m_Param [ PMAX_FRAC ] = %f\n m_Param [ PGRAV ] = %f\n m_Param [ PGROUND_SLOPE ] = %f\n m_Param [ PFORCE_MIN ] = %f\n m_Param [ PFORCE_MAX ] = %f\n m_Param [ PFORCE_FREQ ] = %f\n m_Toggle [ PWRAP_X ] = %i\n m_Toggle [ PWALL_BARRIER ] = %i\n m_Toggle [ PLEVY_BARRIER ] = %i\n m_Toggle [ PDRAIN_BARRIER ] = %i\n m_Param [ PSTAT_NBRMAX ] = %f\n m_Param [ PSTAT_SRCHMAX ] = %f\n m_Vec [ PPOINT_GRAV_POS ].Set ( %f, %f, %f )\n m_Vec [ PPLANE_GRAV_DIR ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_POS ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_RATE ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_ANG ].Set ( %f, %f, %f )\n m_Vec [ PEMIT_DANG ].Set ( %f, %f, %f )\n // Default sim config\n m_Toggle [ PRUN ] = %i\n m_Param [ PGRIDSIZE ] = %f\n m_Vec [ PVOLMIN ].Set ( %f, %f, %f )\n m_Vec [ PVOLMAX ].Set ( %f, %f, %f )\n m_Vec [ PINITMIN ].Set ( %f, %f, %f )\n m_Vec [ PINITMAX ].Set ( %f, %f, %f )\n m_Param [ PFORCE_MIN ] = %f\n m_Param [ PFORCE_FREQ ] = %f\n m_Param [ PGROUND_SLOPE ] = %f\n ",
                           m_Time,
                           m_DT,
                           m_Param [ PSIMSCALE ],
                           m_Param [ PGRID_DENSITY ],
                           m_Param [ PVISC ],
                           m_Param [ PRESTDENSITY ],
                           m_Param [ PSPACING ],
                           m_Param [ PMASS ],
                           m_Param [ PRADIUS ],
                           m_Param [ PDIST ],
                           m_Param [ PSMOOTHRADIUS ],
                           m_Param [ PINTSTIFF ],
                           m_Param [ PEXTSTIFF ],
                           m_Param [ PEXTDAMP ],
                           m_Param [ PACCEL_LIMIT ],
                           m_Param [ PVEL_LIMIT ],
                           m_Param [ PMAX_FRAC ],
                           m_Param [ PGRAV ],
                           m_Param [ PGROUND_SLOPE ],
                           m_Param [ PFORCE_MIN ],
                           m_Param [ PFORCE_MAX ],
                           m_Param [ PFORCE_FREQ ],
                           pwrapx, pwall_barrier, plevy_barrier, pdrain_barrier,
                           m_Param [ PSTAT_NBRMAX ],
                           m_Param [ PSTAT_SRCHMAX ],
                           point_grav_pos.x, point_grav_pos.y, point_grav_pos.z,
                           pplane_grav_dir.x, pplane_grav_dir.y, pplane_grav_dir.z,
                           pemit_pos.x, pemit_pos.y, pemit_pos.z,
                           pemit_rate.x, pemit_rate.y, pemit_rate.z,
                           pemit_ang.x, pemit_ang.y, pemit_ang.z,
                           pemit_dang.x, pemit_dang.y, pemit_dang.z,
                           // Default sim config
                           prun,
                           m_Param [ PGRIDSIZE ],
                           pvolmin.x, pvolmin.y, pvolmin.z,
                           pvolmax.x, pvolmax.y, pvolmax.z,
                           pinitmin.x, pinitmin.y, pinitmin.z,
                           pinitmax.x, pinitmax.y, pinitmax.z,
                           m_Param [ PFORCE_MIN ],
                           m_Param [ PFORCE_FREQ ],
                           m_Param [ PGROUND_SLOPE ]
                          );

    if (m_FParams.debug>1) std::cout << "\nvoid FluidSystem::WriteSimParams (const char * relativePath ) wrote file "<< SimParams_file_path <<"\t"<<
              "ret = " << ret << "\n" << std::flush;
    fclose(SimParams_file);
    return;
}





// from Morphogenesis
void FluidSystem::SavePointsVTP2 ( const char * relativePath, int frame ){// uses vtk library to write binary vtp files
    // based on VtpWrite(....)demo at https://vtk.org/Wiki/Write_a_VTP_file  (30 April 2009)
    // and on https://lorensen.github.io/VTKExamples/site/Cxx/IO/WriteVTP/   (post vtk-8.90.9)

    // Header information:  ?? how can this be added ??
    //  A) fparams & fgenome
    //  B) header of the.csv file, giving sizes of arrays.

    // points, vertices & lines
    // points & vertices = FPOS 3df
    vtkSmartPointer<vtkPoints> points3D = vtkSmartPointer<vtkPoints>::New();                           // Points3D
	vtkSmartPointer<vtkCellArray> Vertices = vtkSmartPointer<vtkCellArray>::New();                     // Vertices

    for ( unsigned int i = 0; i < mMaxPoints; ++i )
	{
		vtkIdType pid[1];
		//Point P = Model.Points[i];
        Vector3DF* Pos = getPos(i);
		pid[0] = points3D->InsertNextPoint(Pos->x, Pos->y, Pos->z);
		Vertices->InsertNextCell(1,pid);
	}
    // edges = FELASTIDX [0]current index uint                                                         // Lines
    vtkSmartPointer<vtkCellArray> Lines = vtkSmartPointer<vtkCellArray>::New();
    uint *ElastIdx;
    float *ElastIdxPtr;
    for ( unsigned int i = 0; i < mMaxPoints; ++i )
	{
        ElastIdx = getElastIdx(i);
        //ElastIdxPtr = (float*)ElastIdx;
        for(int j=0; j<(BONDS_PER_PARTICLE ); j++) {
            int secondParticle = ElastIdx[j * DATA_PER_BOND];
            int bond = ElastIdx[j * DATA_PER_BOND +2];          // NB [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index
            if (bond==0 || bond==UINT_MAX) secondParticle = i;                    // i.e. if [2]restlength, then bond is broken, therefore bond to self.
            vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
            line->GetPointIds()->SetId(0,i);
            line->GetPointIds()->SetId(1,secondParticle);
            Lines->InsertNextCell(line);
        }
	}

    ///////////////////////////////////////////////////////////////////////////////////////////////////// Particle Data

    // FELASTIDX bond data, float and uint vtkDataArrays, stored in particles
    vtkSmartPointer<vtkUnsignedIntArray> BondsUIntData = vtkSmartPointer<vtkUnsignedIntArray>::New();
    BondsUIntData->SetNumberOfComponents(3);
	BondsUIntData->SetName("curr_idx, particle ID, bond index");

    vtkSmartPointer<vtkFloatArray> BondsFloatData = vtkSmartPointer<vtkFloatArray>::New();
    BondsFloatData->SetNumberOfComponents(6);
	BondsFloatData->SetName("elastic limit, restlength, modulus, damping coeff, stress integrator");


    for ( unsigned int i = 0; i < mMaxPoints; ++i )
	{
        ElastIdx = getElastIdx(i);                     // FELASTIDX[BONDS_PER_PARTICLE]  [0]current index uint, [5]particle ID uint, [6]bond index uint
        ElastIdxPtr = (float*)ElastIdx;                // FELASTIDX[BONDS_PER_PARTICLE]  [1]elastic limit float, [2]restlength float, [3]modulus float, [4]damping coeff float,
        for(int j=0; j<(BOND_DATA); j+=DATA_PER_BOND) {
            BondsUIntData->InsertNextTuple3(ElastIdx[j], ElastIdx[j+5], ElastIdx[j+6]);
            BondsFloatData->InsertNextTuple6(ElastIdxPtr[j+1], ElastIdxPtr[j+2], ElastIdxPtr[j+3], ElastIdxPtr[j+4], ElastIdxPtr[j+7], 0);
        }
    }
    //BondsUIntData->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
    //BondsFloatData->SetNumberOfComponents(BONDS_PER_PARTICLE *4);


    // FVEL 3df,
    Vector3DF* Vel;
    vtkSmartPointer<vtkFloatArray> fvel = vtkSmartPointer<vtkFloatArray>::New();
    fvel->SetNumberOfComponents(3);
	fvel->SetName("FVEL");
    for(unsigned int i=0;i<mMaxPoints;i++){
        Vel = getVel(i);
        fvel->InsertNextTuple3(Vel->x,Vel->y,Vel->z);
    }
    fvel->SetNumberOfComponents(BONDS_PER_PARTICLE *3);


/*    // FVEVAL 3df,
    Vector3DF* Veval;
    vtkSmartPointer<vtkFloatArray> fveval = vtkSmartPointer<vtkFloatArray>::New();
    fvel->SetNumberOfComponents(3);
	fvel->SetName("FVEVAL");
    for(unsigned int i=0;i<mMaxPoints;i++){
        Veval = getVeval(i);
        fveval->InsertNextTuple3(Veval->x,Veval->y,Veval->z);
    }
    fveval->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
*/

/*    // FFORCE 3df,
    Vector3DF* Force;
    vtkSmartPointer<vtkFloatArray> fforce = vtkSmartPointer<vtkFloatArray>::New();
    fforce->SetNumberOfComponents(3);
	fforce->SetName("FFORCE");
    for(unsigned int i=0;i<mMaxPoints;i++){
        Force = getForce(i);
        fforce->InsertNextTuple3(Force->x,Force->y,Force->z);
    }
    fforce->SetNumberOfComponents(BONDS_PER_PARTICLE *3);
*/


/*    // FPRESS f,
    float* Pres;
    vtkSmartPointer<vtkFloatArray> fpres = vtkSmartPointer<vtkFloatArray>::New();
    fpres->SetNumberOfComponents(1);
	fpres->SetName("FPRESS");
    for(unsigned int i=0;i<mMaxPoints;i++){
        Pres = getPres(i);
        fpres->InsertNextTuple(Pres);
    }
*/

/*    // FDENSITY f,
    float* Dens;
    vtkSmartPointer<vtkFloatArray> fdens = vtkSmartPointer<vtkFloatArray>::New();
    fdens->SetNumberOfComponents(1);
	fdens->SetName("FDENSITY");
    for(unsigned int i=0;i<mMaxPoints;i++){
        Dens = getDensity(i);
        fdens->InsertNextTuple(Dens);
    }
*/

    // FAGE ushort,
    unsigned int* age = getAge(0);
    vtkSmartPointer<vtkUnsignedIntArray> fage = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fage->SetNumberOfComponents(1);
	fage->SetName("FAGE");
    for(unsigned int i=0;i<mMaxPoints;i++){
        fage->InsertNextValue(age[i]);
    }

    // FCLR uint,
    unsigned int* color = getClr(0);
    vtkSmartPointer<vtkUnsignedIntArray> fcolor = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fcolor->SetNumberOfComponents(1);
	fcolor->SetName("FCLR");
    for(unsigned int i=0;i<mMaxPoints;i++){
        fcolor->InsertNextValue(color[i]);
    }

    // FGCELL	uint,

    // FPARTICLEIDX uint[BONDS_PER_PARTICLE *2],


    // FPARTICLE_ID  uint,
    unsigned int* pid = getParticle_ID(0);
    vtkSmartPointer<vtkUnsignedIntArray> fpid = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fpid->SetNumberOfComponents(1);
	fpid->SetName("FPARTICLE_ID");
    for(unsigned int i=0;i<mMaxPoints;i++){
        fpid->InsertNextValue(pid[i]);
    }

    // FMASS_RADIUS uint (holding modulus 16bit and limit 16bit.),
    unsigned int* Mass_Radius = getMass_Radius(0);
    uint mass, radius;
    vtkSmartPointer<vtkUnsignedIntArray> fmass_radius = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fmass_radius->SetNumberOfComponents(2);
	fmass_radius->SetName("FMASS_RADIUS");
    for(unsigned int i=0;i<mMaxPoints;i++){
        if(Mass_Radius[i]==0){   mass = 0; }else{  mass = Mass_Radius[i]; }
        radius = mass >> 16;
        mass = mass & TWO_POW_16_MINUS_1;
        fmass_radius->InsertNextTuple2(mass,radius);
    }

    // FNERVEIDX uint,
    unsigned int* nidx = getNerveIdx(0);
    vtkSmartPointer<vtkUnsignedIntArray> fnidx = vtkSmartPointer<vtkUnsignedIntArray>::New();
    fnidx->SetNumberOfComponents(1);
	fnidx->SetName("FNERVEIDX");
    for(unsigned int i=0;i<mMaxPoints;i++){
        fnidx->InsertNextValue(nidx[i]);
    }

    // FCONC float[NUM_TF].                                                                                     // commented out until Matt's edit FCONC uint->foat is merged
    vtkSmartPointer<vtkFloatArray> fconc[NUM_TF];
    char buf_conc[256];
    for (int a=0; a<NUM_GENES; a++){
        fconc[a] = vtkSmartPointer<vtkFloatArray>::New();
        fconc[a]->SetNumberOfComponents(1);
        sprintf ( buf_conc, "FCONC_%i",a);
        fconc[a]->SetName(buf_conc);
    }
    float *conc;
    for ( unsigned int i = 0; i < NUM_GENES; ++i ){
        conc = getConc(i);
        for(int j=0; j<mMaxPoints; j++)    fconc[i]->InsertNextValue(conc[j]);                              // now have one array for each column of fepigen
    }

    // FEPIGEN uint[NUM_GENES] ... make an array of arrays
    vtkSmartPointer<vtkUnsignedIntArray> fepigen[NUM_GENES];
    char buf_epigen[256];
    for (int a=0; a<NUM_GENES; a++){
        fepigen[a] = vtkSmartPointer<vtkUnsignedIntArray>::New();
        fepigen[a]->SetNumberOfComponents(1);
        sprintf ( buf_epigen, "FEPIGEN_%i",a);
        fepigen[a]->SetName(buf_epigen);
    }
    unsigned int *epigen;
    for ( unsigned int i = 0; i < NUM_GENES; ++i ){
        epigen = getEpiGen(i);
        for(int j=0; j<mMaxPoints; j++)    fepigen[i]->InsertNextValue(epigen[j]);                              // now have one array for each column of fepigen
    }


    // F_TISSUE_TYPE  uint,
    unsigned int tissueType;
    vtkSmartPointer<vtkUnsignedIntArray> ftissue = vtkSmartPointer<vtkUnsignedIntArray>::New();
    ftissue->SetNumberOfComponents(1);
	ftissue->SetName("F_TISSUE_TYPE");
    unsigned int *epigen_[NUM_GENES];
    for ( unsigned int i = 0; i < NUM_GENES; ++i )  epigen_[i] = getEpiGen(i);

    for(unsigned int i=0;i<mMaxPoints;i++){
        if      (epigen_[9][i] >0/*bone*/)      tissueType =9;
        else if (epigen_[6][i] >0/*tendon*/)    tissueType =6;
        else if (epigen_[7][i] >0/*muscle*/)    tissueType =7;
        else if (epigen_[10][i]>0/*elast lig*/) tissueType =10;
        else if (epigen_[8][i] >0/*cartilage*/) tissueType =8;
        else                                    tissueType =0;
        ftissue->InsertNextValue(tissueType);
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // POLYDATA
	vtkSmartPointer<vtkPolyData> polydata = vtkPolyData::New();                                        // polydata
	polydata->SetPoints(points3D);
	//polydata->SetVerts(Vertices);
    polydata->SetLines(Lines);


    //if (m_FParams.debug>1)cout << "\nStarting writing bond data to polydata\n" << std::flush;
    polydata->GetCellData()->AddArray(BondsUIntData);
    polydata->GetCellData()->AddArray(BondsFloatData);
    //polydata->GetPointData()->AddArray(BondsUIntData);
    //polydata->GetPointData()->AddArray(BondsFloatData);
    polydata->GetPointData()->AddArray(fage);
    polydata->GetPointData()->AddArray(fcolor);
    polydata->GetPointData()->AddArray(fpid);
    polydata->GetPointData()->AddArray(fmass_radius);
    polydata->GetPointData()->AddArray(fnidx);

    for(int i=0;i<NUM_TF; i++)      polydata->GetPointData()->AddArray(fconc[i]);
    for(int i=0;i<NUM_GENES; i++)   polydata->GetPointData()->AddArray(fepigen[i]);

    polydata->GetPointData()->AddArray(ftissue);

    //if (m_FParams.debug>1)cout << "\nFinished writing bond data to polydata\n" << std::flush;

    // WRITER
	vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();       // writer
    char buf[256];
    frame += 100000;                                                                                              // ensures numerical and alphabetic order match of filenames
    sprintf ( buf, "%s/particles_pos_vel_color%04d.vtp", relativePath, frame );
	writer->SetFileName(buf);
	writer->SetInputData(polydata);
    //writer->SetDataModeToAscii();
    writer->SetDataModeToAppended();    // prefered, produces a human readable header followed by a binary blob.
    //writer->SetDataModeToBinary();
	writer->Write();

	//if (m_FParams.debug>1)cout << "\nFinished writing vtp file " << buf << "." << endl;
	//if (m_FParams.debug>1)cout << "\tmMaxPoints: " << mMaxPoints << endl;
}
