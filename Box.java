 import java.awt.*;
 import javax.swing.*;
 import javax.swing.border.*;
 import java.util.ArrayList;
 

//the playing field
public class Box
  extends JPanel
{
  
  private ArrayList<Ball> balls=new ArrayList<>();
  private int width;
  private int height;
  private int delay;

  
  
  //does everything
  public Box(int height, int width, int nCircles, int delay){

    this.height=height;
    this.width=width;
    this.delay=delay;
    
    this.setPreferredSize(new Dimension(width,height));
    this.setBackground(Color.white);
    this.setBorder(new LineBorder(Color.red,2));
    
    
    
    for(int i=0;i<nCircles;i++)
     addBall();
    
    //paintComponent(Graphics g);
    //this.setVisible(true);
    
  }
  
  public void paintComponent(Graphics g){
   //super.paintComponent(g) in this case would allow you to use the class
    //'paintComponent' from JPanel, without it all your balls' trajectories would be traced
   int i=0;
   //super.paintComponent(g);
   while(i<balls.size()){
     balls.get(i).paint(g);
     i++;
   }
   
     
  }
  
  //adds a new ball with random characteristics
  public void addBall(){
    
    Vector position=new Vector(Math.random()*width,Math.random()*height);//position has private access in ball, otherwise non-static context
    Vector velocity=new Vector(Math.random()*width/200,Math.random()*height/200);//inte de snabbaste bollarna
    Color color=new Color((float)Math.random(),
                     (float)Math.random(),
                     (float)Math.random());
    
    double size=Math.random()*Math.PI*width*height/10000;
   
    Ball ball=new Ball(position,velocity,color,size,this);
    balls.add(ball);
    
  }
  //guess
  public int getHeight(){
    
    return this.height;
  }
  
  public int getWidth(){
    
    return this.width;
  }
  
  
  
    public ArrayList<Ball> getBalls(){
    return balls;
    }
  
  //moves the balls according to their velocity vectors
  public void step(){//tick is delay
   Ball ball;
   int i=0;
   //how far should this loop go?
   
     while (i<balls.size()){
     ball=balls.get(i);
     ball.move(this.delay);
    
       i++;
   }
   repaint();
   //balls.forEach(ball->ball.move(tick)); won't work due to my compiler
    
  }
  
    
}

 