from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from pydantic import BaseModel
from mAINidentfy import predecir_imagen
from fanucpy import Robot





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
robot = Robot(
    robot_model="Fanuc",
    host="10.0.0.9",
    port=18735,
    ee_DO_type="RDO",
    ee_DO_num=7,
)
robot.connect()



#PREDICTION OF THE IMAGE
class ImagenRuta(BaseModel):
    ruta_imagen: str

@app.post("/identificar/")
async def identificar(entrada: ImagenRuta):
    ruta_imagen = entrada.ruta_imagen

    try:
        resultado, probabilidad = predecir_imagen(ruta_imagen)

        return {
            "RESULTADO": resultado,
            "PROBABILIDAD": f"{probabilidad * 100:.2f}"
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code="404", detail="IMAGEN NO ENCONTRADA")
    
    except Exception as e:
        raise HTTPException(status_code="500", detail="ERRor en el proceso")



## STATUS OF THE ROBOT
async def is_robot_moving():
    try:
        current_pos = robot.get_curjpos()
        time.sleep(0.1)  # Wait and check again
        new_pos = robot.get_curjpos()

        return current_pos != new_pos  # Returns True if moving, False if still
    except Exception as e:
        print(f"Error checking robot status: {e}")
        return False
    

@app.get("/robotstatus/")
@app.post("/robotstatus/")
async def robotstatus():
    try:
        status = await is_robot_moving()
        print(status)
        return {"robot_moving": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

## for tests
# def robotpos():

#     robotmuving = robot.get_curjpos()
#     robot.disconnect()
#     return print(robotmuving)

#robotpos()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





