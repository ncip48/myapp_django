from django.db import models

# Create your models here.
class StudentStress(models.Model):
    # define columns
    anxiety_level = models.IntegerField()
    self_esteem = models.IntegerField()
    mental_health_history = models.IntegerField()
    depression = models.IntegerField()
    headache = models.IntegerField()
    blood_pressure = models.IntegerField()
    sleep_quality = models.IntegerField()
    breathing_problem = models.IntegerField()
    noise_level = models.IntegerField()
    living_conditions = models.IntegerField()
    safety = models.IntegerField()
    basic_needs = models.IntegerField()
    academic_performance = models.IntegerField()
    study_load = models.IntegerField()
    teacher_student_relationship = models.IntegerField()
    future_career_concerns = models.IntegerField()
    social_support = models.IntegerField()
    peer_pressure = models.IntegerField()
    extracurricular_activities = models.IntegerField()
    bullying = models.IntegerField()
    stress_level = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        # define table name
        db_table = 'student_stress'