#include "mjpc/tasks/simple_car/simple_car.h"

#include <cmath>
#include <cstdio>
#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>


namespace mjpc {

std::string SimpleCar::XmlPath() const {
  return GetModelPath("simple_car/task.xml");
}

std::string SimpleCar::Name() const { return "SimpleCar"; }

float speed = 0.0f;  



void SimpleCar::drawCircle(float radius, int segments) {
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; i++) {
        float angle = 2 * M_PI * i / segments;
        glVertex2f(radius * cos(angle), radius * sin(angle));
    }
    glEnd();
}

void SimpleCar::drawTicks(float radius, int tickCount) {
    float angleStep = 180.0f / tickCount;  
    for (int i = 0; i < tickCount; i++) {
        float angle = i * angleStep * M_PI / 180.0f;
        float tickLength = 0.02f;
        float x1 = radius * cos(angle);
        float y1 = radius * sin(angle);
        float x2 = (radius - tickLength) * cos(angle);
        float y2 = (radius - tickLength) * sin(angle);
        
        glBegin(GL_LINES);
        glVertex2f(x1, y1);
        glVertex2f(x2, y2);
        glEnd();
    }
}


void SimpleCar::drawPointer(float angle) {
    glBegin(GL_LINES);
    glVertex2f(0.0f, 0.0f);  
    glVertex2f(0.8f * cos(angle), 0.8f * sin(angle));  
    glEnd();
}


void SimpleCar::drawNumber(float radius, int number, float angle) {
    char buffer[10];
    snprintf(buffer, sizeof(buffer), "%d", number);
    glRasterPos2f(radius * cos(angle), radius * sin(angle));
    for (int i = 0; buffer[i] != '\0'; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, buffer[i]);
    }
}


void SimpleCar::drawDashboard(float* dashboard_pos, float speed_ratio) {
   glClear(GL_COLOR_BUFFER_BIT);

   
    glPushMatrix();
    glTranslatef(dashboard_pos[0], dashboard_pos[1], dashboard_pos[2]);

    
    glColor3f(0.1f, 0.1f, 0.1f);  
    drawCircle(0.6f, 100);

    
    glColor3f(1.0f, 1.0f, 1.0f);  
    drawTicks(0.5f, 10);  

    
    float pointerAngle = (90.0f - (180.0f * speed_ratio)) * M_PI / 180.0f;  
    glColor3f(1.0f, 0.0f, 0.0f);  
    drawPointer(pointerAngle);

   
    for (int i = 0; i <= 10; i++) {
        float angle = (90.0f - 18.0f * i) * M_PI / 180.0f;
        drawNumber(0.45f, i, angle);
    }

    
    glColor3f(0.9f, 0.9f, 0.9f);
    glRasterPos2f(0.0f, -0.7f);
    const char* unit = "km/h";
    for (int i = 0; unit[i] != '\0'; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, unit[i]);
    }

    
    glColor3f(0.9f, 0.9f, 0.9f);
    char speedStr[50];
    snprintf(speedStr, sizeof(speedStr), "%.1f", speed);
    glRasterPos2f(-0.1f, 0.0f);
    for (int i = 0; speedStr[i] != '\0'; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, speedStr[i]);
    }

    
    glPopMatrix();
    glutSwapBuffers();
}



void SimpleCar::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                     double* residual) const {
  
  residual[0] = data->qpos[0] - data->mocap_pos[0];  
  residual[1] = data->qpos[1] - data->mocap_pos[1];  

  
  residual[2] = data->ctrl[0];  
  residual[3] = data->ctrl[1];  
}


void SimpleCar::TransitionLocked(mjModel* model, mjData* data) {
  
  double car_pos[2] = {data->qpos[0], data->qpos[1]};
  
  
  double goal_pos[2] = {data->mocap_pos[0], data->mocap_pos[1]};
  
 
  double car_to_goal[2];
  mju_sub(car_to_goal, goal_pos, car_pos, 2);
  
  
  if (mju_norm(car_to_goal, 2) < 0.2) {
    absl::BitGen gen_;
    data->mocap_pos[0] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[1] = absl::Uniform<double>(gen_, -2.0, 2.0);
    data->mocap_pos[2] = 0.01;  
  }
}



void SimpleCar::ModifyScene(const mjModel* model, const mjData* data,
                            mjvScene* scene) const {

    
    static double fuel_capacity = 100.0;   
    static double fuel_used = 0.0;    


    double pos_x = data->qpos[0];
    double pos_y = data->qpos[1];


    double vel_x = data->qvel[0];
    double vel_y = data->qvel[1];


    double acc_x = data->qacc[0];
    double acc_y = data->qacc[1];


    double* car_velocity = SensorByName(model, data, "car_velocity");
    double speed_ms = car_velocity ? mju_norm3(car_velocity) : 0.0;


    const int BAR_LEN = 30;
    const double max_speed_ref = 5.0;   
    double rpm_ratio = speed_ms / max_speed_ref;
    if (rpm_ratio > 1.0) rpm_ratio = 1.0;
    if (rpm_ratio < 0.0) rpm_ratio = 0.0;

    int filled = static_cast<int>(rpm_ratio * BAR_LEN);

    char rpm_bar[BAR_LEN + 1];
    for (int i = 0; i < BAR_LEN; i++) {
        rpm_bar[i] = (i < filled) ? '#' : ' ';
    }
    rpm_bar[BAR_LEN] = '\0';


    double dt = model->opt.timestep;


    double throttle = data->ctrl[0];


    const double fuel_coeff = 0.2;


    fuel_used += fuel_coeff * std::abs(throttle) * dt;


    if (fuel_used > fuel_capacity) {
        fuel_used = fuel_capacity;
    }
    double fuel_left = fuel_capacity - fuel_used;
    double fuel_percent = (fuel_left / fuel_capacity) * 100.0;


    if (fuel_percent < 0.0) fuel_percent = 0.0;
    if (fuel_percent > 100.0) fuel_percent = 100.0;




    printf(
            "\rPos(%.2f, %.2f) | "
            "Vel(%.2f, %.2f) | "
            "Acc(%.2f, %.2f) | "
            "Fuel %3.0f%%"
            "RPM [%s",
            pos_x, pos_y,
            vel_x, vel_y,
            acc_x, acc_y,
            fuel_percent,
            rpm_bar
    );


    fflush(stdout);




   
  int car_body_id = mj_name2id(model, mjOBJ_BODY, "car");
  if (car_body_id < 0) return;  
  
 

  if (!car_velocity) return;  
  
 

  double speed_kmh = speed_ms * 3.6;  
  
  
  double* car_pos = data->xpos + 3 * car_body_id;
  
 
  float dashboard_pos[3] = {
    static_cast<float>(car_pos[0]),
    static_cast<float>(car_pos[1] ),  
    static_cast<float>(car_pos[2] + 0.3f)   
  };



  const float gauge_scale = 6.0f;  
  const float max_speed_kmh = 10.0f; 


  float speed_ratio = static_cast<float>(speed_kmh) / max_speed_kmh;
  if (speed_ratio > 1.0f) speed_ratio = 1.0f;
  
 
  double angle_x = 90.0 * 3.14159 / 180.0;  
  double cos_x = cos(angle_x);
  double sin_x = sin(angle_x);
  double mat_x[9] = {
    1, 0,      0,
    0, cos_x, -sin_x,
    0, sin_x,  cos_x
  };
  
  double angle_z = -90.0 * 3.14159 / 180.0;  
  double cos_z = cos(angle_z);
  double sin_z = sin(angle_z);
  double mat_z[9] = {
    cos_z, -sin_z, 0,
    sin_z,  cos_z, 0,
    0,      0,     1
  };
  
  
  double dashboard_rot_mat[9];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      dashboard_rot_mat[i*3 + j] = 0;
      for (int k = 0; k < 3; k++) {
        dashboard_rot_mat[i*3 + j] += mat_z[i*3 + k] * mat_x[k*3 + j];
      }
    }
  }



    
  if (scene->ngeom < scene->maxgeom) {
    mjvGeom* geom = scene->geoms + scene->ngeom;
    
    
    geom->type = mjGEOM_CYLINDER;
      geom->size[0] = geom->size[1] = 0.15f * gauge_scale;
      geom->size[2] = 0.005f * gauge_scale;
      
    
    geom->pos[0] = dashboard_pos[0];
    geom->pos[1] = dashboard_pos[1];
    geom->pos[2] = dashboard_pos[2];
    
   
    for (int j = 0; j < 9; j++) {
      geom->mat[j] = static_cast<float>(dashboard_rot_mat[j]);
    }
    
    
    geom->rgba[0] = 0.9f;
    geom->rgba[1] = 0.9f;
    geom->rgba[2] = 0.9f;
    geom->rgba[3] = 0.05f;  
    scene->ngeom++;
  }

    
    const int kMaxTick = 4;
    const int kTickCount = kMaxTick + 1;

    const int ARC_SEG = 20;
    for (int s = 0; s < ARC_SEG; s++) {
        if (scene->ngeom >= scene->maxgeom) break;

        
        float t0 = (ARC_SEG <= 1) ? 0.0f : (float)s / (float)(ARC_SEG - 1);

        float angle_deg = 180.0f - 180.0f * t0;
        float rad = angle_deg * 3.14159f / 180.0f;

        float arc_r = 0.145f * gauge_scale;
        float arc_y = dashboard_pos[1] - arc_r * (float)cos(rad);
        float arc_z = dashboard_pos[2] + arc_r * (float)sin(rad);

        
        double rot_deg = angle_deg - 90.0;
        double rr = rot_deg * 3.14159 / 180.0;
        double c = cos(rr), si = sin(rr);

        double zrot[9] = { c, -si, 0,
                           si,  c, 0,
                           0,   0, 1 };

        double matd[9];
        for (int r = 0; r < 3; r++) {
            for (int cc = 0; cc < 3; cc++) {
                matd[r*3 + cc] = 0.0;
                for (int k = 0; k < 3; k++) {
                    matd[r*3 + cc] += dashboard_rot_mat[r*3 + k] * zrot[k*3 + cc];
                }
            }
        }

        
        float rcol = (t0 > 0.8f) ? 1.0f : (0.0f + 1.25f * t0);   
        float gcol = 0.0f;                                       
        float bcol = (t0 > 0.8f) ? 0.20f : 1.0f;                 


        



       
        mjvGeom* g = scene->geoms + scene->ngeom;

        mjtNum size[3] = {
                (mjtNum)(0.0025f * gauge_scale),
                (mjtNum)(0.0060f * gauge_scale),   
                (mjtNum)(0.0025f * gauge_scale)
        };

        mjtNum pos[3] = {
                (mjtNum)dashboard_pos[0],
                (mjtNum)arc_y,
                (mjtNum)arc_z
        };

        mjtNum mat9[9];
        for (int j = 0; j < 9; j++) mat9[j] = (mjtNum)matd[j];

        float rgba[4] = { rcol, gcol, bcol, 0.85f };

        mjv_initGeom(g, mjGEOM_BOX, size, pos, mat9, rgba);
       

        scene->ngeom++;
    }




    for (int i = 0; i < kTickCount; i++) {
        if (scene->ngeom >= scene->maxgeom) break;

        int tick_value = i;

        
        float tick_angle_deg = 180.0f - (180.0f * tick_value / kMaxTick);
        float rad_tick_angle = tick_angle_deg * 3.14159f / 180.0f;

        
        float full_len = ((tick_value % 5 == 0) ? 0.030f : 0.020f) * gauge_scale;
        float half_len = full_len * 0.5f;




        float tick_radius_outer = 0.135f * gauge_scale;
        float tick_radius_center = tick_radius_outer - half_len;

        mjvGeom* geom = scene->geoms + scene->ngeom;
        geom->type = mjGEOM_BOX;
       
        geom->size[0] = 0.001f * gauge_scale;
        geom->size[1] = half_len * 1.5f;             
        geom->size[2] = 0.001f * gauge_scale; 
       


        float tick_y = dashboard_pos[1] - tick_radius_center * cos(rad_tick_angle);
        float tick_z = dashboard_pos[2] + tick_radius_center * sin(rad_tick_angle);

        geom->pos[0] = dashboard_pos[0];
        geom->pos[1] = tick_y;
        geom->pos[2] = tick_z;

        
        double tick_rot_angle = tick_angle_deg - 90.0;
        double rad_tick_rot = tick_rot_angle * 3.14159 / 180.0;
        double cos_t = cos(rad_tick_rot);
        double sin_t = sin(rad_tick_rot);

        double tick_rot_mat[9] = {
                cos_t, -sin_t, 0,
                sin_t,  cos_t, 0,
                0,      0,     1
        };

        double tick_mat[9];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                tick_mat[r*3 + c] = 0;
                for (int k = 0; k < 3; k++) {
                    tick_mat[r*3 + c] += dashboard_rot_mat[r*3 + k] * tick_rot_mat[k*3 + c];
                }
            }
        }

        for (int j = 0; j < 9; j++) {
            geom->mat[j] = static_cast<float>(tick_mat[j]);
        }
  
        geom->rgba[0] = 0.2f;
        geom->rgba[1] = 0.2f;
        geom->rgba[2] = 0.2f;
        geom->rgba[3] = 0.5f;
        scene->ngeom++;

        
        if (scene->ngeom >= scene->maxgeom) break;

        mjvGeom* label_geom = scene->geoms + scene->ngeom;
        label_geom->type = mjGEOM_LABEL;
        label_geom->size[0] = label_geom->size[1] = label_geom->size[2] = 0.05f * gauge_scale;


        float label_radius = 0.18f * gauge_scale;

        label_geom->pos[0] = dashboard_pos[0];
        label_geom->pos[1] = dashboard_pos[1] - label_radius * cos(rad_tick_angle);
        label_geom->pos[2] = dashboard_pos[2] + label_radius * sin(rad_tick_angle);
        
        label_geom->rgba[0] = 0.8f;
        label_geom->rgba[1] = 0.8f;
        label_geom->rgba[2] = 0.8f;
        label_geom->rgba[3] = 1.0f;

        std::snprintf(label_geom->label,
                      sizeof(label_geom->label),
                      "%d",
                      tick_value);

        scene->ngeom++;
    }

    
  if (scene->ngeom < scene->maxgeom) {
    mjvGeom* geom = scene->geoms + scene->ngeom;
    geom->type = mjGEOM_BOX;
      
      geom->size[0] = 0.002f * gauge_scale;
      geom->size[1] = 0.080f * gauge_scale;
      geom->size[2] = 0.002f * gauge_scale;
      

    
    
    float angle = 180.0f - 180.0f * speed_ratio;  
    float rad_angle = angle * 3.14159f / 180.0f;
    
    
      float pointer_y = dashboard_pos[1] - 0.0275f * gauge_scale * cos(rad_angle);
      float pointer_z = dashboard_pos[2] + 0.0275f * gauge_scale * sin(rad_angle);


    geom->pos[0] = dashboard_pos[0];
    geom->pos[1] = pointer_y;
    geom->pos[2] = pointer_z;
    
    
    double pointer_angle = angle - 90.0;  
    double rad_pointer_angle = pointer_angle * 3.14159 / 180.0;
    double cos_p = cos(rad_pointer_angle);
    double sin_p = sin(rad_pointer_angle);
    double pointer_rot_mat[9] = {
      cos_p, -sin_p, 0,
      sin_p,  cos_p, 0,
      0,      0,     1
    };
    
    
    double temp_mat[9];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        temp_mat[i*3 + j] = 0;
        for (int k = 0; k < 3; k++) {
          temp_mat[i*3 + j] += dashboard_rot_mat[i*3 + k] * pointer_rot_mat[k*3 + j];
        }
      }
    }
    
    for (int i = 0; i < 9; i++) {
      geom->mat[i] = static_cast<float>(temp_mat[i]);
    }

    geom->rgba[0] = 1.0f;  
    geom->rgba[1] = 0.5f;  
    geom->rgba[2] = 0.5f;  
    geom->rgba[3] = 1.0f;
    scene->ngeom++;
  }
  
  
  if (scene->ngeom < scene->maxgeom) {
    mjvGeom* geom = scene->geoms + scene->ngeom;
    geom->type = mjGEOM_SPHERE;
      geom->size[0] = geom->size[1] = geom->size[2] = 0.006f * gauge_scale;

      geom->pos[0] = dashboard_pos[0];
    geom->pos[1] = dashboard_pos[1];
    geom->pos[2] = dashboard_pos[2];
    // 应用仪表盘旋转矩阵
    for (int j = 0; j < 9; j++) {
      geom->mat[j] = static_cast<float>(dashboard_rot_mat[j]);
    }
    geom->rgba[0] = 0.8f;  
    geom->rgba[1] = 0.8f;
    geom->rgba[2] = 0.8f;
    geom->rgba[3] = 1.0f;
    scene->ngeom++;
  }
  
  
  if (scene->ngeom < scene->maxgeom) {
    mjvGeom* geom = scene->geoms + scene->ngeom;
    geom->type = mjGEOM_LABEL;
    geom->size[0] = geom->size[1] = geom->size[2] = 0.08f;
    geom->pos[0] = dashboard_pos[0];
    geom->pos[1] = dashboard_pos[1];
    geom->pos[2] = dashboard_pos[2] + 0.02f;  
    
    geom->rgba[0] = 0.9f;  
    geom->rgba[1] = 0.9f;
    geom->rgba[2] = 0.9f;
    geom->rgba[3] = 1.0f;
    
    char speed_label[50];
    std::snprintf(speed_label, sizeof(speed_label), "%.1f", speed_kmh);
    std::strncpy(geom->label, speed_label, sizeof(geom->label) - 1);
    geom->label[sizeof(geom->label) - 1] = '\0';
    scene->ngeom++;
  }
  
  
  if (scene->ngeom < scene->maxgeom) {
    mjvGeom* geom = scene->geoms + scene->ngeom;
    geom->type = mjGEOM_LABEL;
    geom->size[0] = geom->size[1] = geom->size[2] = 0.05f;
    geom->pos[0] = dashboard_pos[0];
    geom->pos[1] = dashboard_pos[1];
    geom->pos[2] = dashboard_pos[2] - 0.06f;  
    
    geom->rgba[0] = 0.8f;  
    geom->rgba[1] = 0.8f;
    geom->rgba[2] = 0.8f;
    geom->rgba[3] = 1.0f;
    
    std::strncpy(geom->label, "km/h", sizeof(geom->label) - 1);
    geom->label[sizeof(geom->label) - 1] = '\0';
    scene->ngeom++;
  }
}

}  // namespace mjpc
